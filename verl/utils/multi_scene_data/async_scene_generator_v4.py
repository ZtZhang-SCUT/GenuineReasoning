import asyncio
import json
import logging
import threading
import uuid
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import aiohttp
import heapq
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from verl.utils.multi_scene_data.prompt_constants import MULTI_ROUND_PROMPT_TEMPLATE, MULTI_ROUND_PROMPT_TEMPLATE_EN, PROMPT_TEMPLATE, PROMPT_TEMPLATE_EN

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneGeneratorCallback:
    """场景生成回调处理器，仅负责副作用：解析、保存、日志等，不返回结果"""

    def __init__(self, save_dir: str = "./"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def parse_output(self, raw_output: str) -> Optional[Dict[str, str]]:
        """解析模型输出为结构化结果"""
        try:
            text = raw_output.strip()
            # 在去除 ```json 后、解析前，加入：
            text = text.replace('\x0c', '\\f').replace('\x08', "\\b")  # 把 form feed 换回 \f
            # print(text)
            if text.startswith("```json"):
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()

            text = self._fix_invalid_json(text)
            parsed = json.loads(text)

            if parsed and "new_question" in parsed and "solution" in parsed:
                return {
                    "question": parsed["new_question"],
                    "solution": parsed["solution"],
                    "explanation": parsed.get("explanation", ""),
                }
            else:
                logger.warning(f"Missing required fields in output: {text[:200]}...")
                return None
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f"JSON parse failed: {e}")
            return None

    def _fix_invalid_json(self, text: str) -> str:
        import re
        def repl(match):
            prefix, bad_char = match.group(1), match.group(2)
            return prefix + ("\\" + bad_char if len(prefix) % 2 == 1 else bad_char)
        return re.sub(r'(\\+)(?!["\\/bfnrtu])(.?)', repl, text)

    def on_generation_complete(
        self,
        messages: List[Dict[str, str]],
        raw_output: str,
        info: Dict[str, Any],
        parsed_result: Optional[Dict[str, str]],
    ) -> None:
        """生成完成后调用，执行保存等操作"""
        save_file = info.get("__save_file__", "")
        is_save = info.get("__is_save__", False)
        if is_save:
            if not save_file:
                ts = info.get("timestamp", str(uuid.uuid4()))
                save_file = self.save_dir / f"scene_{ts}.jsonl"

            save_path = Path(save_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                "messages": messages,
                "raw_output": raw_output,
                "parsed_result": parsed_result,
                "info": {k: v for k, v in info.items() if not k.startswith("__")},
            }

            with save_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if parsed_result:
                logger.info(f"✅ Saved result to {save_file}")
            else:
                logger.warning(f"⚠️ Saved failed result to {save_file}")

        else:
            logger.info(f"Save file is not provided, no need to saved result")

class SceneGeneratorScheduler:
    def __init__(
        self,
        api_urls: List[str],
        api_keys: List[str],
        model_name: str = "your-large-model",
        timeout: float = 30,
        max_concurrent: int = 10,
        max_retries: int = 3,
        temperature: float = 0.8,
        request_interval: float = 0.1,
        callback: Optional[SceneGeneratorCallback] = None,
    ):
        self.model_name = model_name
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_interval = request_interval
        self.callback = callback or SceneGeneratorCallback()

        self.api_configs = [[0, url, key] for url, key in zip(api_urls, api_keys)]
        heapq.heapify(self.api_configs)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # 重试装饰器配置
        self._retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
            before_sleep=before_sleep_log(logging.getLogger(), logging.INFO)
        )
        self.background_tasks = set()

    async def _call_api_async_impl(
        self, api_url: str, api_key: str, messages: List[Dict[str, str]]
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "x-request-id": uuid.uuid4().hex,
            }
        payload = {
            "model": self.model_name, 
            "messages": messages, 
            "temperature": self.temperature
            }
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(api_url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()

    @property
    def call_api_with_retry(self):
        # 返回一个装饰了重试逻辑的函数
        return self._retry_decorator(self._call_api_async_impl)
    
    async def _call_api_async(self, api_url: str, api_key: str, messages: List[Dict[str, str]]) -> str:
        return await self.call_api_with_retry(api_url, api_key, messages)
    
    async def _generate_single_async(
        self, messages: List[Dict[str, str]], api_url: str, api_key: str
    ) -> str:
        async with self._semaphore:
            if self.request_interval > 0:
                await asyncio.sleep(self.request_interval)
            return await self._call_api_async(api_url, api_key, messages)

    async def generate_one_with_callback(
        self, messages: List[Dict[str, str]], info: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """生成一个结果，触发 callback，并返回解析后的结果"""
        # 选择服务器
        config = self.api_configs[0]
        api_url, api_key = config[1], config[2]
        config[0] += 1
        heapq.heapreplace(self.api_configs, config)

        raw_output = None
        try:
            raw_output = await self._generate_single_async(messages, api_url, api_key)
            parsed = self.callback.parse_output(raw_output)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            parsed = None

        # 触发 callback（副作用）
        self.callback.on_generation_complete(messages, raw_output or "", info, parsed)
        return parsed

    def submit_generation(self, messages: List[Dict[str, str]], info: Dict[str, Any]):
        """异步提交，不等待结果"""
        task = asyncio.create_task(self._fire_and_forget(messages, info))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _fire_and_forget(self, messages: List[Dict[str, str]], info: Dict[str, Any]):
        await self.generate_one_with_callback(messages, info)


class SceneGeneratorManager:
    def __init__(
        self,
        api_urls: List[str],
        api_keys: List[str],
        model_name: str = "your-large-model",
        timeout: float = 30,
        max_concurrent: int = 10,
        max_retries: int = 3,
        temperature: float = 0.8,
        request_interval: float = 1,
    ):
        self.loop = None
        self.scheduler: Optional[SceneGeneratorScheduler] = None
        self.ready = threading.Event()

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.scheduler = SceneGeneratorScheduler(
                    api_urls=api_urls,
                    api_keys=api_keys,
                    model_name=model_name,
                    timeout=timeout,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                    temperature=temperature,
                    request_interval=request_interval,
                )
            finally:
                self.ready.set()
            self.loop.run_forever()

        threading.Thread(target=run_loop, daemon=True).start()
        self.ready.wait()

    def _format_previous_aug(self, previous_aug_questions: List[str]) -> str:
        """格式化之前生成的问题"""
        if not previous_aug_questions: return ""
        formatted = []
        for i, aug_question in enumerate(previous_aug_questions, 1):
            formatted.append(f"{i}. {aug_question}")
        return "\n".join(formatted)
    
    def _build_messages(self, original_question: str, previous: List[str]) -> List[Dict[str, str]]:
        if previous:
            aug = self._format_previous_aug(previous)
            prompt = MULTI_ROUND_PROMPT_TEMPLATE_EN.format(
                previous_aug=aug, original_question=original_question
            )
        else:
            # import random
            # scenarios = ["医学", "金融", "农业", "日常生活", "税务"]
            # random.shuffle(scenarios)
            prompt = PROMPT_TEMPLATE_EN.format(
                # scenarios=scenarios, 
                original_question=original_question
            )
        return [{"role": "user", "content": prompt}]

    def generate_single(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        save_file: Optional[str] = None,
        **extra_info,
    ) -> Optional[Dict[str, str]]:
        """同步生成单个结果，自动触发 callback 并返回结果"""
        messages = self._build_messages(original_question, previous_aug_questions)
        info = {
            "__save_file__": save_file, 
            "__is_save__": True if save_file else False, 
            **extra_info
            }

        future = asyncio.run_coroutine_threadsafe(
            self.scheduler.generate_one_with_callback(messages, info),
            self.loop,
        )
        return future.result()

    def submit_generation(
        self,
        original_question: str,
        previous_aug_questions: List[str],
        save_file: Optional[str] = None,
        **extra_info,
    ):
        """异步提交生成任务，不等待结果"""
        messages = self._build_messages(original_question, previous_aug_questions)
        info = {
            "__save_file__": save_file, 
            "__is_save__": True if save_file else False, 
            **extra_info
            }
        asyncio.run_coroutine_threadsafe(
            self.scheduler.submit_generation(messages, info),
            self.loop,
        )

    def generate_batch(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, str]]]:
        """同步批量生成，每个任务可指定 save_file"""
        async def _batch():
            coros = []
            for task in tasks:
                messages = self._build_messages(
                    task["original_question"], task.get("previous_aug_questions", [])
                )
                info = {"__save_file__": task.get("save_file"), 
                        "__is_save__": True if task.get("save_file") else False,
                        **task.get("extra_info", {})}
                coros.append(self.scheduler.generate_one_with_callback(messages, info))
            return await asyncio.gather(*coros)

        future = asyncio.run_coroutine_threadsafe(_batch(), self.loop)
        return future.result()
    
if __name__ == "__main__":
    callback = SceneGeneratorCallback(save_dir="./my_outputs")
    mgr = SceneGeneratorManager(
        # api_urls=["http://152.136.41.186:30131/v1/chat/completions"],
        api_urls=["http://172.31.128.29:8001/v1/chat/completions"],   # 172.31.128.29  172.31.196.23
        api_keys=["sk-kQKMTKyEQA7X6eZ_AR72Cqvb2o7NgskyrKG9UdPqUPD8zu-_HeVs5Nie0_M"],
        # model_name="deepseek-v3.1t",
        timeout=180,
        # model_name="llama3.1-8b",
        model_name="qwen3-235b-a22b",
        temperature=0
        # callback=callback,
    )

    # result = mgr.generate_single(
    #     original_question="如何计算复利？",
    #     previous_aug_questions=[],
    #     save_file="./my_outputs/compound_interest.jsonl"
    # )
    # print("Result:", result)

    tasks = [
        {
            "original_question": "如果一个水池有两个水管，水管A单独注水需要3小时，水管B单独注水需要6小时。那么两个水管一起注水需要多少时间才能注满水池？",
            "previous_aug_questions": [],
            "save_file": "./batch/q1.jsonl"
        },
        # {
        #     "original_question": "如果一个水池有两个水管，水管A单独注水需要3小时，水管B单独注水需要6小时。那么两个水管一起注水需要多少时间才能注满水池？",
        #     "previous_aug_questions": ["历史生成问题处在税务的场景中"],
        #     "save_file": "./batch/q2.jsonl"
        # }
    ]
    results = mgr.generate_batch(tasks)
    print("Batch results:", results)