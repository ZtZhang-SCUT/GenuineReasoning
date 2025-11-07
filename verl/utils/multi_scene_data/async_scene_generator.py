import asyncio
import json
import logging
import threading
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, before_sleep_log
from collections import deque
import heapq
from functools import partial
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
# 问题生成任务指令
你是一个专业的问题生成器，给定学生未正确解答的原始问题，执行以下流程生成新问题：
1. 知识点与场景解析：提取原始问题的核心数学知识点及其所属场景（上下文）；
2. 场景迁移与复杂度提升：将提取的知识点迁移至全新复杂现实场景（如{scenarios}等），确保场景与原始问题无关联；
3. 新问题构建：严格遵循以下约束条件：
   - 数学结构一致性：保留原始问题的核心数学逻辑（如公式形式、推理框架、求解步骤）；
   - 难度适度增强：通过增加现实场景约束条件提升难度，确保答案唯一且明确；
   - 解答完整性：配套完整解题过程，关键数学表达式需兼容LaTeX解析规范，并把最终的答案放在 \\boxed() 中。

# 输出要求
仅以JSON格式返回以下字段（嵌入```json代码块中，无额外文本）：
```json
{{
"new_question": "迁移后场景的完整问题描述",
"solution": "含LaTeX格式表达式的详细解题步骤及最终答案",
"explanation": "知识点对应关系、场景迁移依据及难度提升设计说明"
}}
```

## 输入信息
原始问题：{original_question}
"""

MULTI_ROUND_PROMPT_TEMPLATE = """
# 问题增强任务：生成差异化新问题
给定学生未正确解答的原始问题及该问题的历史增强记录，生成一个全新问题，要求如下：

## 核心约束
1. 知识点迁移：严格保留原始问题的核心数学结构（如公式形式、推理逻辑、求解步骤）；
2. 场景创新：新问题的现实场景必须与历史增强问题的场景完全不同（历史场景见下方「历史增强记录」）；
3. 解答完整性：需附带含LaTeX格式的详细解题步骤，确保最终答案唯一，并把最终的答案放在 \\boxed() 中。

## 历史增强记录
{previous_aug}

## 输出要求
仅返回JSON格式内容（嵌入```json代码块，无额外文本），包含：
- new_question：新问题完整描述（需体现独特场景）
- solution：解题步骤+LaTeX公式答案
- explanation：说明新问题与原始问题的知识点对应关系，及与历史增强问题的场景差异

```json
{{
"new_question": "",
"solution": "",
"explanation": ""
}}
```

## 输入信息
原始问题：{original_question}
"""

class SceneGeneratorScheduler:
    def __init__(
        self,
        api_urls: List[str],  # 支持多个API服务器
        api_keys: List[str],  # 对应的API密钥
        model_name: str = "your-large-model",
        timeout: Union[int, float] = 30,
        max_concurrent: int = 10,
        max_retries: int = 3,
        temperature: float = 0.8,
        request_interval: float = 0.1,
    ):
        """
        Args:
            api_urls: 多个API服务器地址列表
            api_keys: 对应的API密钥列表
            model_name: 模型名称
            timeout: 请求超时（秒）
            max_concurrent: 最大并发请求数
            max_retries: 失败重试次数
            temperature: 模型温度
            request_interval: 请求间隔（秒）
        """
        self.model_name = model_name
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_interval = request_interval
        
        # 负载均衡：使用最少请求的服务器
        self.api_configs = [[0, url, key] for url, key in zip(api_urls, api_keys)]
        heapq.heapify(self.api_configs)
        
        # 信号量用于控制并发数
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # 重试装饰器配置
        self._retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
            before_sleep=before_sleep_log(logging.getLogger(), logging.INFO)
        )
        
        # 后台任务管理
        self.background_tasks = set()

    def submit_generation(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        request_id: Optional[str] = None,
        save_file: Optional[str] = None,
        callback: Optional[callable] = None
    ):
        """提交生成请求，不等待完成，完成后调用callback"""
        task = asyncio.create_task(
            self._submit_generation_and_callback(
                original_question, previous_aug_questions, request_id, save_file, callback
            )
        )
        
        # "fire-and-forget" 后台任务
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_generation_and_callback(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        request_id: Optional[str],
        save_file: Optional[str],
        callback: Optional[callable]
    ):
        """提交生成请求，等待完成并执行回调"""
        # 选择服务器（负载均衡）
        selected_config = self.api_configs[0]
        api_url, api_key = selected_config[1], selected_config[2]
        selected_config[0] += 1  # 增加计数
        heapq.heapreplace(self.api_configs, selected_config)
        
        result, exception = None, None
        try:
            result = await self._generate_single_async(
                original_question, previous_aug_questions, api_url, api_key
            )
            
            # 保存原始数据
            if save_file and result:
                await self._dump_to_jsonl_async({
                    "original_question": original_question,
                    "previous_aug_questions": previous_aug_questions,
                    "result": result,
                    "request_id": request_id
                }, save_file)
                
        except Exception as e:
            exception = e
            logger.error(f"Generation failed for question '{original_question[:50]}...': {e}")

        # 执行回调
        if callback:
            try:
                callback(result, exception)
            except Exception as e:
                logger.error(f"Callback failed: {e}")

    async def _generate_single_async(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        api_url: str,
        api_key: str
    ) -> Optional[Dict[str, str]]:
        """异步生成单个场景变体"""
        prompt = self._construct_prompt(original_question, previous_aug_questions)
        
        async with self._semaphore:  # 控制并发数
            if self.request_interval > 0:
                await asyncio.sleep(self.request_interval)
            
            raw_output = await self._call_api_async(api_url, api_key, prompt)
            parsed = self._parse_json(raw_output)

            if parsed and "new_question" in parsed and "solution" in parsed:
                result = {
                    "question": parsed["new_question"],
                    "solution": parsed["solution"],
                    "explanation": parsed.get("explanation", ""),
                }
                return result
            else:
                logger.warning(f"Failed to parse output: {raw_output[:200]}...")
                return None

    async def _call_api_async_impl(self, api_url: str, api_key: str, prompt: str) -> str:
        """异步调用大模型 API"""
        # raise RuntimeError("something wrong")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()

    @property
    def call_api_with_retry(self):
        # 返回一个装饰了重试逻辑的函数
        return self._retry_decorator(self._call_api_async_impl)
    
    async def _call_api_async(self, api_url: str, api_key: str, prompt: str) -> str:
        return await self.call_api_with_retry(api_url, api_key, prompt)

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """安全解析 JSON，容忍 Markdown 代码块"""
        try:
            if text.strip().startswith("```json"):
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            
            text = self._fix_invalid_json(text)
            return json.loads(text)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None

    # def _fix_invalid_json(self, text: str) -> str:
    #     """修复 JSON 中的非法转义字符"""
    #     import re
        
    #     # 修复过度转义的 LaTeX 符号
    #     text = re.sub(r'\\\\([\$\\%{}])', r'\\\1', text)
    #     # 修复 \$ 为 $
    #     text = text.replace(r'\$', '$')
    #     # 修复其他非法转义
    #     text = re.sub(r'\\([^"\\/bfnrtu])', r'\\\1', text)
        
    #     return text

    def _fix_invalid_json(self, text: str) -> str:
        """
        修复非法 JSON 转义字符：
        - 仅当 \ 后不是合法 JSON 转义字符时修复；
        - 若该反斜杠前已有偶数个反斜杠（已被正确转义）则不修复；
        - 仅在奇数个反斜杠结尾时修复为双反斜杠。
        """

        def repl(match):
            prefix = match.group(1)
            bad_char = match.group(2)
            # 计算反斜杠数量
            backslash_count = len(prefix)
            # 如果是偶数个反斜杠，说明上一个反斜杠已经被转义，不修复
            if backslash_count % 2 == 0:
                return prefix + bad_char
            # 如果是奇数个反斜杠，说明这是非法转义 -> 加倍
            return prefix + "\\" + bad_char

        # 匹配：连续反斜杠 + 一个非合法转义字符
        pattern = r'(\\+)(?!["\\/bfnrtu])(.?)'
        return re.sub(pattern, repl, text)

    def _construct_prompt(
        self,
        original_question: str,
        previous_aug_questions: List[str],
    ) -> str:
        """构建多轮对话提示词"""
        if previous_aug_questions:
            previous_aug = "\n".join(previous_aug_questions)
            return MULTI_ROUND_PROMPT_TEMPLATE.format(
                previous_aug=previous_aug,
                original_question=original_question
            )
        else:
            return PROMPT_TEMPLATE.format(
                scenarios=self._shuffle_scenarios(),
                original_question=original_question
            )

    def _format_previous_aug(self, previous_aug_questions: List[str]) -> str:
        """格式化之前生成的问题"""
        if not previous_aug_questions: return ""
        formatted = []
        for i, aug_question in enumerate(previous_aug_questions, 1):
            formatted.append(f"{i}. {aug_question}")
        return "\n".join(formatted)

    def _shuffle_scenarios(self) -> List[str]:
        """随机打乱场景"""
        import random
        scenarios = ["医学", "金融", "农业", "日常生活", "税务"]
        random.shuffle(scenarios)
        return scenarios

    async def _dump_to_jsonl_async(self, data: Dict[str, Any], save_file: str):
        """异步保存数据到 JSONL 文件"""
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with save_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def _generate_batch_async(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, str]]]:
        """异步批量生成"""
        results = [None] * len(tasks)
        sem = asyncio.Semaphore(self.max_concurrent)  # 每个批次的并发控制
        
        async def process_task(i, task):
            async with sem:
                result = await self._generate_single_async(
                    task["original_question"],
                    task["previous_aug_questions"],
                    task["api_url"],
                    task["api_key"]
                )
                results[i] = result
                return result
        
        tasks_coroutines = [
            process_task(i, {
                **task,
                "api_url": self.api_configs[0][1],
                "api_key": self.api_configs[0][2]
            }) for i, task in enumerate(tasks)
        ]
        
        await asyncio.gather(*tasks_coroutines)
        return results
    
    async def _submit_generation_semaphore(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        save_file: Optional[str],
        callback: Optional[callable]
    ):
        """提交生成请求并等待完成"""
        done = asyncio.Event()
        result = [None]
        exception_occurred = [None]
        
        def callback_wrapper(res, exc):
            result[0] = res
            exception_occurred[0] = exc
            done.set()
        
        self.submit_generation(
            original_question=original_question,
            previous_aug_questions=previous_aug_questions,
            save_file=save_file,
            callback=callback_wrapper
        )
        
        await done.wait()
        
        if exception_occurred[0]:
            raise exception_occurred[0]
        
        return result[0]


class SceneGeneratorManager:
    """SceneGeneratorManager 管理多个 SceneGenerator 实例"""

    def __init__(
        self,
        api_urls: List[str],
        api_keys: List[str],
        model_name: str = "your-large-model",
        timeout: Union[int, float] = 30,
        max_concurrent: int = 10,
        max_retries: int = 3,
        temperature: float = 0.8,
        request_interval: float = 0.1,
    ):
        """初始化 SceneGeneratorManager"""
        self.api_urls = api_urls
        self.api_keys = api_keys
        self.model_name = model_name
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_interval = request_interval

        # 在单独的线程中初始化异步调度器
        self.scene_generator: SceneGeneratorScheduler = None
        self.scene_generator_exception: Exception = None
        self.scene_generator_loop = None
        self.scene_generator_ready = threading.Event()
        self.scene_generator_thread = threading.Thread(target=self._init_scene_generator, daemon=True)
        self.scene_generator_thread.start()
        self.scene_generator_ready.wait()

    def _init_scene_generator(self):
        """在单独线程中初始化异步调度器"""
        self.scene_generator_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.scene_generator_loop)

        try:
            self.scene_generator = SceneGeneratorScheduler(
                api_urls=self.api_urls,
                api_keys=self.api_keys,
                model_name=self.model_name,
                timeout=self.timeout,
                max_concurrent=self.max_concurrent,
                max_retries=self.max_retries,
                temperature=self.temperature,
                request_interval=self.request_interval,
            )
        except Exception as e:
            logger.exception(f"scene generator init error: {e}")
            self.scene_generator_exception = e
        finally:
            self.scene_generator_ready.set()
        
        self.scene_generator_loop.run_forever()

    def submit_generation(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        save_file: Optional[str] = None,
        callback: Optional[callable] = None
    ):
        """提交生成请求并等待完成"""
        assert self.scene_generator is not None, "scene generator is not initialized."
        
        future = asyncio.run_coroutine_threadsafe(
            self.scene_generator._submit_generation_semaphore(
                original_question, 
                previous_aug_questions, 
                save_file, 
                callback
            ),
            self.scene_generator_loop,
        )
        return future.result()

    def generate_single(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        save_file: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """同步生成单个场景变体"""
        return self.submit_generation(
            original_question=original_question,
            previous_aug_questions=previous_aug_questions,
            save_file=save_file
        )

    def generate_batch(
        self,
        questions_data: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, str]]]:
        """同步批量生成"""
        assert self.scene_generator is not None, "scene generator is not initialized."
        
        future = asyncio.run_coroutine_threadsafe(
            self.scene_generator._generate_batch_async([
                {
                    "original_question": item["original_question"],
                    "previous_aug_questions": item["previous_aug_questions"]
                }
                for item in questions_data
            ]),
            self.scene_generator_loop
        )
        return future.result()

    def generate_and_save(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        save_file: str
    ) -> Optional[Dict[str, str]]:
        """同步版本的 generate_and_save（保持与原接口兼容）"""
        return self.generate_single(original_question, previous_aug_questions, save_file)


# ============ 使用示例 ============

def example_usage():
    # 初始化管理器
    manager = SceneGeneratorManager(
        api_urls=["http://localhost:8001/v1/chat/completions"],#  "http://152.136.41.186:30131/v1/chat/completions"],
        api_keys=["sk-kQKMTKyEQA7X6eZ_AR72Cqvb2o7NgskyrKG9UdPqUPD8zu-_HeVs5Nie0_M"],# "sk-kQKMTKyEQA7X6eZ_AR72Cqvb2o7NgskyrKG9UdPqUPD8zu-_HeVs5Nie0_M"],
        model_name="llama3.1-8b",
        temperature=0,
        max_concurrent=5,
        max_retries=3
    )
    
    # 单个生成
    result = manager.generate_single(
        "如果一个水池有两个水管，水管A单独注水需要3小时，水管B单独注水需要6小时。那么两个水管一起注水需要多少时间才能注满水池？",
        ["之前的问题1", "之前的问题2"],
        "save_file.jsonl"
    )
    print(f"Single generation result: {result}")
    
    # 批量生成
    questions_data = [
        {
            "original_question": "如果一个水池有两个水管，水管A单独注水需要3小时，水管B单独注水需要6小时。那么两个水管一起注水需要多少时间才能注满水池？",
            "previous_aug_questions": []
        },
        {
            "original_question": "如果一个水池有两个水管，水管A单独注水需要3小时，水管B单独注水需要6小时。那么两个水管一起注水需要多少时间才能注满水池？", 
            "previous_aug_questions": ["已生成问题"]
        }
    ]
    results = manager.generate_batch(questions_data)
    print(f"Batch generation results: {results}")
    
    # 兼容原接口
    # result = manager.generate_and_save(
    #     "原始问题",
    #     ["之前的问题1", "之前的问题2"], 
    #     "save_file.jsonl"
    # )
    # print(f"Compatible interface result: {result}")

if __name__ == "__main__":
    example_usage()