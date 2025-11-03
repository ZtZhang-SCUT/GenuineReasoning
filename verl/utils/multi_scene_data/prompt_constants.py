# PROMPT_TEMPLATE = """
# 你是一个专业的问题生成器，给定学生未正确解答的原始问题，执行以下流程生成新问题：
# 1. 知识点与场景解析：提取原始问题的核心数学知识点及其所属场景（上下文）；
# 2. 场景迁移：将提取的知识点迁移至全新复杂现实场景，确保场景与原始问题无关联；
# 3. 新问题构建：严格遵循以下约束条件：
#    - 数学结构一致性：保留原始问题的核心数学逻辑（如公式形式、推理框架、求解步骤）；
#    - 解答完整性：需附带含LaTeX格式的详细解题步骤，确保最终答案唯一，并把最终的答案放在 \\boxed() 中。

# # 输出要求
# 仅以JSON格式返回以下字段（嵌入```json代码块中，无额外文本）：
# ```json
# {{
# "new_question": "迁移后场景的完整问题描述",
# "solution": "含LaTeX格式表达式的详细解题步骤及最终答案",
# "explanation": "知识点对应关系、场景迁移依据"
# }}
# ```

# ## 输入信息
# 原始问题：{original_question}
# """

# PROMPT_TEMPLATE = """
# 你是一个专业的问题生成器。给定一个学生未正确解答的原始问题，请基于其核心数学知识点（如公式和求解步骤），将其迁移到一个与原始场景无关的、真实且具有挑战性的全新现实场景中，生成一个结构一致但情境全新的问题。

# 要求：
# - 新问题保留原始问题的数学本质，场景需真实、新颖且与原始问题所处场景无关联；
# - 提供符合 LaTeX 格式的完整解题步骤，确保答案唯一且为数值，并将最终答案用 \\boxed{{}} 包裹；
# - 仅输出以下 JSON 结构，不要包含其它无关文本。

# 请严格按照以下格式返回结果：
# ```json
# {{
#   "new_question": "问题在新场景下的完整描述",
#   "solution": "符合 LaTeX 格式的解题步骤及最终答案，答案用 \\boxed{{}} 包裹",
#   "explanation": "说明原始与新问题的知识点对应关系及场景迁移依据"
# }}
# ```

# 原始问题：{original_question}
# """


PROMPT_TEMPLATE = """
你是一个专业的问题生成器。给定一个学生未正确解答的原始问题，请基于其核心数学知识点（如公式和求解步骤），将其迁移到一个与原始场景无关的、真实且具有挑战性的全新现实场景中，生成一个结构一致但情境全新的问题。

要求：
- 新问题保留原始问题的数学本质，场景需真实、新颖且与原始问题所处场景无关联；
- 提供含LaTeX格式的详细解题步骤，确保最终答案唯一，并把最终答案放在 \\boxed{{}} 中。
- 仅输出以下 JSON 结构，不要包含其它无关内容。

请严格按照以下格式返回结果：
```json
{{
  "new_question": "问题在新场景下的完整描述",
  "solution": "解题步骤及最终答案，答案用 \\boxed{{}} 包裹",
  "explanation": "说明原始与新问题的知识点对应关系及场景迁移依据"
}}```

原始问题：{original_question}
"""

PROMPT_TEMPLATE_EN = """
You are a professional problem generator. Given an original problem that a student answered incorrectly, your task is to create a new problem based on the same core mathematical concepts (such as formulas and solution steps), but set in a completely new, realistic, and challenging context that is unrelated to the original scenario.

Requirements:
- The new problem must preserve the mathematical essence of the original one, but its real-world scenario should be novel, realistic, and independent of the original context.
- Provide a **detailed solution** in LaTeX format, ensuring the **final answer is unique** and enclosed in \\boxed{{}}.
- Output **only** the following JSON structure, with **no extra text** or explanations outside the JSON.

Please strictly follow this format:
```json
{{
  "new_question": "A complete description of the problem in the new scenario",
  "solution": "Step-by-step solution in LaTeX, with the final answer enclosed in \\boxed{{}}",
  "explanation": "An explanation of how the core concepts correspond between the original and new problems, and the rationale behind the scenario transfer"
}}```

Original problem: {original_question}
"""

# 符合LaTeX格式的详细解题步骤及最终答案，答案用 \\boxed{{}} 包裹

# MULTI_ROUND_PROMPT_TEMPLATE = """
# 给定学生未正确解答的原始问题及该问题的历史增强记录，生成一个全新问题，要求如下：

# ## 核心约束
# 1. 知识点迁移：严格保留原始问题的核心数学结构（如公式形式、推理逻辑、求解步骤）；
# 2. 场景创新：新问题的现实场景必须与历史增强问题的场景完全不同（历史场景见下方「历史增强记录」）；
# 3. 解答完整性：需附带含LaTeX格式的详细解题步骤，确保最终答案唯一，并把最终的答案放在 \\boxed() 中。

# ## 历史增强记录
# {previous_aug}

# ## 输出要求
# 仅返回JSON格式内容（嵌入```json代码块，无额外文本），包含：
# - new_question：新问题完整描述（需体现独特场景）
# - solution：解题步骤+LaTeX公式答案
# - explanation：说明新问题与原始问题的知识点对应关系，及与历史增强问题的场景差异

# ```json
# {{
# "new_question": "",
# "solution": "",
# "explanation": ""
# }}
# ```

# ## 输入信息
# 原始问题：{original_question}
# """

# 提供完整的解题过程，使用 LaTeX 格式书写，确保答案唯一且为数值，并将最终答案用 \\boxed{{}} 标注；

MULTI_ROUND_PROMPT_TEMPLATE = """
你是一个专业的问题生成器。给定一个学生未正确解答的原始问题及其历史增强记录，请基于原始问题的核心数学结构（包括公式形式、推理逻辑和求解步骤），生成一个数学本质一致但场景全新的问题。

要求：
- 新问题保留原始问题的数学本质，所生成的新场景需真实、有挑战性，且与原始问题及所有历史增强问题的场景均无关联；
- 提供含LaTeX格式的详细解题步骤，确保最终答案唯一，并把最终答案放在 \\boxed{{}} 中。
- 仅输出以下 JSON 结构，不要包含其它无关文本。

请严格按照以下格式返回结果：
```json
{{
  "new_question": "新场景下的完整问题描述",
  "solution": "符合 LaTeX 格式的解题步骤及最终答案，答案用 \\boxed{{}} 包裹",
  "explanation": "说明与原始问题的知识点对应关系，以及与历史增强问题在场景上的差异"
}}```

原始问题：{original_question}

历史增强记录：
{previous_aug}
"""

MULTI_ROUND_PROMPT_TEMPLATE_EN = """
You are a professional problem generator. Given an original math problem that a student failed to solve correctly, along with its previous enhancement records, generate a new problem that preserves the **core mathematical structure** of the original (including formulas, reasoning logic, and solution steps), but places it in a **completely new and realistic context**.

Requirements:
- The new problem must retain the same mathematical essence as the original, while introducing a **realistic and challenging scenario** that is **unrelated** to both the original problem and all previous enhanced versions.
- Provide a **detailed solution** in LaTeX format, ensuring the **final answer is unique** and enclosed in \\boxed{{}}.
- Output **only** the following JSON structure, with **no extra text** or explanations outside the JSON.

Please follow this exact format:
```json
{{
  "new_question": "The complete problem description set in the new context",
  "solution": "Step-by-step solution in LaTeX format, with the final answer enclosed in \\boxed{{}}",
  "explanation": "An explanation of how the new problem corresponds to the mathematical concept of the original question, and how its context differs from all previous enhanced versions"
}}```

Original problem:
{original_question}

Previous enhancement records:
{previous_aug}
"""