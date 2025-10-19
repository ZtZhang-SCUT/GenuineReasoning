import json
import time
import requests
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

# ==============================
# Scene Generator Class
# ==============================

prompt_template = """
You are a scene rewriting expert. Your task is to rewrite the given "question-answer" pair into an equivalent question in a specified new scene, **while keeping the original question's solution logic and answer completely unchanged**.
"""
PROMPT_TEMPLATE = (
    "你是一个场景改写专家。你的任务是将给定的“问题-答案”对，重写为指定新场景下的等价问题，但**必须保持原始问题的解法逻辑和答案完全不变**。\n\n"
    "规则：\n"
    "1. 仅改变问题的表述上下文（如角色、行业、语言风格、背景故事等）；\n"
    "2. 不得改变问题的数学/逻辑/事实本质；\n"
    "3. 答案必须与原始答案在语义上严格等价；\n"
    "4. 输出必须是合法 JSON，包含 \"question\" 和 \"answer\" 字段。\n\n"
    "示例：\n"
    "原始问题：计算本金1000元，年利率5%，复利3年后的本息和。\n"
    "原始答案：1157.63元\n"
    "目标场景：金融顾问向客户解释\n\n"
    "输出：\n"
    "{{\"question\": \"假设您投资了1000元，年化收益率5%，按复利计算，3年后您能拿回多少钱？\", \"answer\": \"1157.63元\"}}\n\n"
    "现在，请处理以下任务：\n\n"
    "原始问题：{original_question}\n"
    "原始答案：{original_answer}\n"
    "目标场景：{target_scene}\n\n"
    "请输出改写后的 JSON："
)

class SceneGenerator:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "your-large-model",
        timeout: int = 30,
        max_retries: int = 3,
        scenes: List[str] = None,
    ):
        """
        Args:
            api_url: 大模型 API 地址（如 Volcano/DashScope/OpenAI 兼容 endpoint）
            api_key: API 密钥
            model_name: 模型名称
            timeout: 请求超时（秒）
            max_retries: 失败重试次数
            scenes: 默认场景列表，如 ["医学", "金融", "农业", "日常生活", "税务"]
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.scenes = scenes or ["医学", "金融", "农业", "日常生活", "税务"]

        self.prompt_template = PROMPT_TEMPLATE

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str) -> str:
        """调用大模型 API，返回纯文本响应"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # 降低随机性
            "max_tokens": 512,
        }
        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        # 根据实际 API 调整字段，以下为通用格式
        return result["choices"][0]["message"]["content"].strip()

    def _parse_json(self, text: str) -> Optional[Dict[str, str]]:
        """安全解析 JSON，容忍 Markdown 代码块"""
        try:
            # 移除 ```json ... ``` 包裹
            if text.startswith("```"):
                text = text.split("```")[1] if "```" in text else text
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return None

    def _single_generate(
        self,
        original_question: str,
        original_answer: str,
        target_scenes: Optional[List[str]] = None,
    ) -> List[Optional[Dict[str, str]]]:
        """
        为一个问题生成多个场景变体。
        
        Returns:
            List of dicts with keys "question", "answer", or None if failed.
        """
        scenes = target_scenes or self.scenes
        results = []

        for scene in scenes:
            prompt = self.prompt_template.format(
                original_question=original_question,
                original_answer=original_answer,
                target_scene=scene,
            )
            try:
                raw_output = self._call_api(prompt)
                parsed = self._parse_json(raw_output)
                if parsed and "question" in parsed and "answer" in parsed:
                    parsed = {**parsed, "scene": scene}
                    results.append(parsed)
                else:
                    print(f"⚠️ Failed to parse output for scene '{scene}': {raw_output[:100]}...")
                    results.append(None)
            except Exception as e:
                print(f"❌ API call failed for scene '{scene}': {e}")
                results.append(None)
            time.sleep(0.5)  # 避免 QPS 超限

        return results
    
    def generate(
        self,
        question_list: List[str],
        answer_list: List[str],
        target_scenes: Optional[List[str]] = None,
    ):
        """
        为一批问题生成多个场景变体。
        
        Args:
            question_list: List of original questions.
            answer_list: List of original answers.
            target_scenes: Optional list of target scenes. If None, use self.scenes.
        
        Returns:
            List of lists, each inner list contains dicts with keys "question", "answer", "scene", or None if failed.
        """
        assert len(question_list) == len(answer_list), "Questions and answers must match in length."
        all_results = []
        for q, a in zip(question_list, answer_list):
            scene_variants = self._single_generate(q, a, target_scenes)
            all_results.append(scene_variants)
        return all_results


# ==============================
# Integration into RayPPOTrainer
# ==============================

# 在 RayPPOTrainer.__init__ 中添加：
"""
if config.get("scene_generator", None):
    self.scene_generator = SceneGenerator(
        api_url=config.scene_generator.api_url,
        api_key=config.scene_generator.api_key,
        model_name=config.scene_generator.model_name,
        scenes=config.scene_generator.get("scenes", ["医学", "金融", "农业", "日常生活", "税务"]),
    )
else:
    self.scene_generator = None
"""

# 使用示例（在 fit 中）：
"""
if self.scene_generator is not None:
    # 假设 batch.non_tensor_batch["raw_prompt"] 和 ["reference_answer"] 存在
    original_q = batch.non_tensor_batch["raw_prompt"][0]  # 取第一个作为示例
    original_a = batch.non_tensor_batch.get("reference_answer", [""])[0]
    scene_variants = self.scene_generator.generate_scenes(original_q, original_a)
    # scene_variants 是 List[Dict]，可进一步处理为 prompts
"""