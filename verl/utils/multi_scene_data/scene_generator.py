import json
import time
import requests
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from verl.utils.multi_scene_data.process_data import make_map_fn
import os

# ==============================
# Scene Generator Class
# ==============================


# 修改 PROMPT_TEMPLATE 以适应具体需求
# 提示词里面要有占位符用以放置模型感到困惑的题目
# 要求模型先分析该题涉及到的知识点，然后判断问题所处的场景/上下文，接着生成涉及相关知识点但不同场景的新问题，并给出对应解题过程+答案。


# 非填充用的 {} 转义为 {{}}
PROMPT_TEMPLATE = """
你是一个专业的问题生成器，以下是一个学生当前无法答对的问题：
原始问题：{original_question}
请先分析该题涉及到的知识点以及该问题的上下文是属于什么场景，然后将这些知识点迁移到一个全新且复杂的现实场景中，例如：税务、农业、医学、化学、地理等，生成一个新的问题。新的问题必须满足以下要求：
1. 新问题必须涉及与原问题相同的数学结构，但场景必须完全不同；
2. 新问题的难度应适度增加，确保问题有明确、唯一的正确答案；
3. 新问题必须附带完整的解答过程，确保最终答案能通过latex解析。
请以JSON格式输出：
```json
{{
"new_question":  把新问题放在这里, 
"solution": 把解题过程和最终答案放在这里,
"explanation": 说明为什么这么设计问题
}}
```
"""


PROMPT_TEMPLATE = """
# 问题生成任务指令
你是一个专业的问题生成器，给定学生未正确解答的原始问题，执行以下流程生成新问题：
1. 知识点与场景解析：提取原始问题的核心数学知识点及其所属场景（上下文）；
2. 场景迁移与复杂度提升：将提取的知识点迁移至全新复杂现实场景（如税务、农业、医疗、化学、地理等），确保场景与原始问题无关联；
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

def format_previous_aug(previous_aug_questions):
    if not previous_aug_questions: return ""
    formatted = []
    for i, aug_question in enumerate(previous_aug_questions, 1):
        formatted.append(f"{i}. {aug_question}")
    return "\n".join(formatted)

class SceneGenerator:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "your-large-model",
        timeout: int = 30,
        max_retries: int = 3,
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
        # self.scenes = scenes or ["医学", "金融", "农业", "日常生活", "税务"]  # 太生硬

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
            "temperature": 0.8,  # 增加随机性
            # "max_tokens": 512,
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
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            return json.loads(text, strict=False) # text 中可能包含 \n, \r, tab键等特殊字符，导致json解析失败，添加strict=False可避免
        except (json.JSONDecodeError, IndexError):
            return None
        

    # 由于需求变更，暂时注释掉该方法
    def _generate_multi_scene_variant_for_one_question(
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

    def _construct_prompt(
        self,
        original_question: str,
        previous_aug_questions: List[str],
    ) -> str:
        """
        构建多轮对话提示词。
        
        Args:
            original_question: 原始问题
            previous_aug_questions: 之前基于original_question生成的新问题，列表形式
        
        Returns:
            构建好的基于original_question、previous_aug_questions的提示词，提示模型生成与previous_aug_questions不同的新问题
        """
        # assert round == len(previous_aug_questions)+1, f"{round=}, {len(previous_aug_questions)=}"
        if previous_aug_questions:
            # 以适当的方式拼接previous_aug_questions
            previous_aug = format_previous_aug(previous_aug_questions)
            return MULTI_ROUND_PROMPT_TEMPLATE.format(
                previous_aug = previous_aug,
                original_question = original_question
            )
        
        else:
            return PROMPT_TEMPLATE.format(
                original_question = original_question
            )
       
    def generate_and_save(
        self,
        original_question: str,
        previous_aug_questions: list[str],
        save_file: str
    ) -> Optional[Dict[str, str]]:
        """
        为一个问题生成一个OOD场景变体。
        
        Returns:
            Dict with keys "question", "answer", or None if failed.
        """
        prompt = self._construct_prompt(original_question, previous_aug_questions)
        # print(f"[round]: {round} \n[prompt]: {prompt}")
        try:
            raw_output = self._call_api(prompt)
            parsed = self._parse_json(raw_output)

            model_side_data = {
                "prompt": prompt,
                "model_output": raw_output,
                "parsed_output": parsed
            }

            self._dump_to_jsonl(model_side_data, save_file)

            if parsed and "new_question" in parsed and "solution" in parsed:
                result = {
                    "question": parsed["new_question"],
                    "solution": parsed["solution"],
                    "explanation": parsed.get("explanation", ""),
                }
                return result
            else:
                print(f"⚠️ Failed to parse OOD output: {raw_output[:]}")
                return None
        except Exception as e:
            print(f"❌ API call failed for generate_and_save: {e}")
            return None
        

    def _generate_single_ood_scene_variant_for_one_question(
        self,
        original_question: str,
        previous_aug_questions: list[str],
    ) -> Optional[Dict[str, str]]:
        """
        为一个问题生成一个OOD场景变体。
        
        Returns:
            Dict with keys "question", "answer", or None if failed.
        """
        prompt = self._construct_prompt(original_question, previous_aug_questions)
        # print(f"[round]: {round} \n[prompt]: {prompt}")
        try:
            raw_output = self._call_api(prompt)
            parsed = self._parse_json(raw_output)
            if parsed and "new_question" in parsed and "solution" in parsed:
                result = {
                    "question": parsed["new_question"],
                    "solution": parsed["solution"],
                    "explanation": parsed.get("explanation", ""),
                }
                return result
            else:
                print(f"⚠️ Failed to parse OOD output: {raw_output[:]}")
                return None
        except Exception as e:
            print(f"❌ API call failed for OOD scene: {e}")
            return None
    
    # def generate(
    #     self,
    #     question_list: List[str],
    #     answer_list: List[str],
    #     target_scenes: Optional[List[str]] = None,
    # ):
    #     """
    #     为一批问题生成多个场景变体。
        
    #     Args:
    #         question_list: List of original questions.
    #         answer_list: List of original answers.
    #         target_scenes: Optional list of target scenes. If None, use self.scenes.
        
    #     Returns:
    #         List of lists, each inner list contains dicts with keys "question", "answer", "scene", or None if failed.
    #     """
    #     assert len(question_list) == len(answer_list), "Questions and answers must match in length."
    #     all_results = []
    #     for q, a in zip(question_list, answer_list):
    #         scene_variants = self._single_generate(q, a, target_scenes)
    #         all_results.append(scene_variants)
    #     return all_results

    def generate_until_reach_target_num(
        self,
        question: str,
        target_num: int,
    ) -> List[Dict[str, str]]:
        """
        不断生成场景变体直到达到目标数量。
        
        Args:
            question: Original question.
            answer: Original answer.
            target_num: Desired number of unique scene variants.
            target_scenes: Optional list of target scenes. If None, use self.scenes.
        
        Returns:
            List of dicts with keys "question", "answer", "scene".
        """
        unique_results = {}
        
        while len(unique_results) < target_num:
            variant = self._generate_single_ood_scene_variant_for_one_question(question, previous_aug_questions, round)
            if variant and variant["question"] not in unique_results:
                unique_results[variant["question"]] = variant
            if len(unique_results) >= target_num:
                break
        
        return list(unique_results.values())[:target_num]

    def generate_with_history(self, original_question, num_questions=5):
        generated = []  # 存储历史生成的问题
        for i in range(num_questions):
            
            new_item = self._generate_single_ood_scene_variant_for_one_question(original_question, generated, i)
            if new_item:
                generated.append(new_item["question"])
                print(f"生成第{i+1}个问题：{len(generated)}")
            else:
                print("新问题与历史场景重复，重试...")
                i -= 1  # 重试当前轮次
            # try:
            #     # 验证关键字段+场景差异（简单校验：新问题场景不在历史场景摘要中）
            #     if all(key in new_item for key in ["question", "solution", "explanation"]):
            #         generated.append(new_item["question"])
            #         print(f"生成第{i+1}个问题：{len(generated)}")
            #     else:
            #         print("新问题与历史场景重复，重试...")
            #         i -= 1  # 重试当前轮次
            # except Exception as e:
            #     print(f"解析失败：{e}，重试...")
            #     i -= 1  # 重试当前轮次
        return generated
    
    def _dump_to_jsonl(self, item: dict, save_file: str):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    



if __name__ == "__main__":
    # 简单测试
    generator = SceneGenerator(
        api_url="http://152.136.41.186:30131/v1/chat/completions",
        api_key="sk-kQKMTKyEQA7X6eZ_AR72Cqvb2o7NgskyrKG9UdPqUPD8zu-_HeVs5Nie0_M",
        model_name="qwen3-max",
    )
    test_question = "如果一个水池有两个水管，水管A单独注水需要3小时，水管B单独注水需要6小时。那么两个水管一起注水需要多少时间才能注满水池？"
    generated = generator.generate_with_history(test_question, 3)
    for idx, var in enumerate(generated):
        print(f"Variant {idx+1}:")
        print(f"Question: {var}")
    
    # variants = generator.generate_until_reach_target_num(test_question, target_num=1)
    # for idx, var in enumerate(variants):
    #     print(f"Variant {idx+1}:")
    #     print("Question:", var["question"])
    #     print("Answer:", var["answer"])
    #     print("Explanation:", var.get("explanation", ""))
    #     print("-" * 40)
