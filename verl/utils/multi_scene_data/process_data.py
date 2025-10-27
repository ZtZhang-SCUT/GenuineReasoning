from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
import numpy as np
from scipy.optimize import fsolve


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>"
boxed_instruction = "{}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# add a row to each data item that represents a unique id
def make_map_fn(split, question_key, answer_key, do_extract_solution, reward_fn_extraction_type, nothink = False):

    def process_fn(example, idx):
        question = example.pop(question_key)

        if reward_fn_extraction_type == 'answer':
            formatted_question = (instruction_following if not nothink else instruction_following.strip(' <think>')).format(question)
        elif reward_fn_extraction_type == 'boxed':
            formatted_question = boxed_instruction.format(question)
        elif reward_fn_extraction_type == 'none':
            formatted_question = question
        
        # gpqa has this string in the question
        if reward_fn_extraction_type != 'boxed':
            remove_string = "\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end."
            replacement_string = '\n\nPlease reason step-by-step and put your choice letter without any other text with <answer> </answer> in the end.'
            formatted_question = formatted_question.replace(remove_string, replacement_string)

        answer = example.pop(answer_key)
        if do_extract_solution:
            solution = extract_solution(answer)
        else:
            solution = answer
        data_source = example.pop('data_source')
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": formatted_question
            }],
            "problem": question,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                # 'metric': METRIC_MAP[data_source],
            }
        }
        return data

    return process_fn


class ModifiedGaussian:
    def __init__(self, x0=0.5, c=0.1, k=20):
        """
        初始化修正型高斯函数（满足特定端点约束的单峰函数）
        :param x0: 最大值位置（0 < x0 < 1）
        :param c: x=1 时的函数值（0 < c < 1，小正数）
        :param k: 曲线陡峭程度（k > 0，越大曲线越尖）
        """
        assert 0 < x0 < 1, "最大值位置 x0 必须在 (0,1) 之间"
        assert 0 < c < 1, "x=1 处函数值 c 必须在 (0,1) 之间"
        assert k > 0, "陡峭系数 k 必须为正数"
        
        self.x0 = x0  # 最大值位置
        self.c = c    # x=1 处的函数值
        self.k = k    # 控制曲线陡峭程度
        
        # 求解参数 A, b, d 满足约束条件
        self.A, self.b, self.d = self._solve_parameters()
        
    def _solve_parameters(self):
        """通过约束条件求解高斯函数的参数 A, b, d"""
        # 约束方程：
        # 1. f(0) = 0 → A*exp(-k*x0²) + d = 0
        # 2. f(x0) = 1 → A + b*x0 + d = 1
        # 3. f(1) = c → A*exp(-k*(1-x0)²) + b + d = c
        
        def equations(vars):
            A, b, d = vars
            eq1 = A * np.exp(-self.k * self.x0**2) + d  # 等于0
            eq2 = A + b * self.x0 + d - 1               # 等于0
            eq3 = A * np.exp(-self.k * (1 - self.x0)**2) + b + d - self.c  # 等于0
            return [eq1, eq2, eq3]
        
        # 初始猜测值，求解方程组
        A_init, b_init, d_init = 1.0, 0.0, -0.5
        A, b, d = fsolve(equations, [A_init, b_init, d_init])
        return A, b, d
    
    def __call__(self, x):
        """计算函数值（支持单个数值或数组）"""
        x = np.asarray(x)
        # 修正型高斯函数表达式
        gaussian_term = self.A * np.exp(-self.k * (x - self.x0)** 2)
        linear_term = self.b * x + self.d
        f_x = gaussian_term + linear_term
        # 确保值域严格在 [0,1] 内（修正数值误差）
        # return np.clip(f_x, 0.0, 1.0)
        return f_x
