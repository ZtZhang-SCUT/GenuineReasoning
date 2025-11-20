# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the `Text-Only` part of OlympiadBench to parquet format
"""

import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs
from datasets import concatenate_datasets


chinese_answer_type_dict = {
	'Numerical': '数值',
	'Expression': '表达式',
	'Equation': '方程',
	'Interval': '区间'
}
english_answer_type_dict = {
	'Numerical': 'a numerical value',
	'Expression': 'an expression',
	'Equation': 'an equation',
	'Interval': 'an interval'
}

def get_single_answer_type_text(answer_type, is_chinese):
	if '-' in answer_type:	# No need now
		answer_type = answer_type[:answer_type.find('-')]
	for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
		if t in answer_type:
			if is_chinese:
				return chinese_answer_type_dict[t]
			else:
				return english_answer_type_dict[t]
	exit(f'Error parsing answer type {answer_type}!')

def get_answer_type_text(answer_type, is_chinese, multiple_answer):
	if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):	# 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
		full_answer_text = ''
	else:
		if not multiple_answer:
			answer_text = get_single_answer_type_text(answer_type, is_chinese)
			if is_chinese:
				full_answer_text = f'，答案类型为{answer_text}'
			else:
				full_answer_text = f"The answer of The problem should be {answer_text}. "
		else:
			if ',' not in answer_type:	# Same answer type for all answers
				answer_text = get_single_answer_type_text(answer_type, is_chinese)
				if is_chinese:
					full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
				else:
					full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
			else:
				answer_types = answer_type.split(',')
				answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
				if len(set(answer_types)) == 1:
					answer_text = answer_types[0]
					if is_chinese:
						full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
					else:
						full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
				else:
					if is_chinese:
						answer_text = '、'.join(answer_types)
						full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
					else:
						answer_text = ', '.join(answer_types)
						full_answer_text = f'The problem has multiple answers, with the answers in order being {answer_text}. '
	return full_answer_text

class InstructionMaker:
	def __init__(self, ds_part_name):
		self.is_theorem_proving = 'TP' in ds_part_name
		self.is_math = 'maths' in ds_part_name
		self.is_chinese = 'zh' in ds_part_name

	def make_prompt(self, question):
		if self.is_chinese:
			subject_content = '数学' if self.is_math else '物理'
			if self.is_theorem_proving:
				prompt = f'以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。'
			else:
				answer_type_text = get_answer_type_text(question['answer_type'], is_chinese=True, multiple_answer=question['is_multiple_answer'])
				if question['is_multiple_answer']:
					multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
				else:
					multiple_answer_text = '\\boxed{答案}'
				unit_text = ''
				if question['unit']:
					multiple_answer_text += '(单位)'
					unit_text = '，注意答案的单位不要放在\\boxed{}中'
				prompt = f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”显式给出结果{unit_text}。'
		else:
			subject_content = 'Math' if self.is_math else 'Physics'
			if self.is_theorem_proving:
				prompt = f'The following is a theorem proving problem from an International {subject_content} competition. Please use logical reasoning and common theorems to prove the proposition in the problem according to the given requirements. Please use LaTeX format to represent the variables and formulas used in the proof.'
			else:
				if question['is_multiple_answer']:
					multiple_answer_text = '\\boxed{multiple answers connected with commas}'
				else:
					multiple_answer_text = '\\boxed{answer}'
				unit_text = ''
				if question['unit']:
					multiple_answer_text += '(unit)'
					unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
				answer_type_text = get_answer_type_text(question['answer_type'], is_chinese=False, multiple_answer=question['is_multiple_answer'])
				prompt = f'The following is an open-ended problem from an International {subject_content} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'
		return prompt

	def make_input(self, prompt, question_content):
		content = prompt + '\n' + question_content + '\n'
		# Adding the prompt recommended in Deepseek-Math's huggingface repository
		if self.is_chinese:
			content += '请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
		else:
			content += 'Please reason step by step, and put your final answer within \\boxed{}.'
		return content
	
	def get_answer(self, sample):
		return sample["final_answer"][0]
	
	def make_instruction(self, question: dict):
		prompt = self.make_prompt(question)
		if self.is_math:
			input = self.make_input(prompt, question['question'])
		else:
			if 'context' in question.keys() and question['context']: # cannot be null
				input = self.make_input(prompt, question['context']+'\n'+question['question'])
			else:
				input = self.make_input(prompt, question['question'])
		return input
	
	def get_precision(self, sample):
		precision = 1e-8
		answer_type = sample["answer_type"]
		if "Tuple" in answer_type:
			pass
		else:
			if sample["error"]:
				if ',' in sample["error"]:
					precisions = sample["error"].split(',')
					precisions = [float(p) if p else 1e-8 for p in precisions]
				else:
					precision = float(sample["error"])
			else:
				pass
		return precision
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_dir", default="~/data/math")
	parser.add_argument("--hdfs_dir", default=None)

	args = parser.parse_args()

	# 'lighteval/MATH' is no longer available on huggingface.
	# Use mirror repo: DigitalLearningGmbH/MATH-lighteval
	data_source = "Hothan/OlympiadBench"
	print(f"Loading the {data_source} dataset from huggingface...", flush=True)
	def process(data_source, ds_part_name, split):
		dataset = datasets.load_dataset(data_source, ds_part_name, trust_remote_code=True)
		dataset = dataset[split]
		instruction_maker = InstructionMaker(ds_part_name)
		# add a row to each data item that represents a unique id
		def make_map_fn(split, data_source):
			def process_fn(example, idx):
				instruction = instruction_maker.make_instruction(example)
				solution = example["final_answer"][0]
				precision = instruction_maker.get_precision(example)
				data = {
					"data_source": data_source,
					"prompt": [{"role": "user", "content": instruction}],
					"ability": "math",
					"reward_model": {"style": "rule", "ground_truth": solution},
					"extra_info": {"split": "test", "index": idx, "precision": precision},
				}
				return data

			return process_fn

		dataset = dataset.map(function=make_map_fn(split, data_source), with_indices=True)
		return dataset

	dataset = []
	for ds_part_name in ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP"]:
		dataset.append(process(data_source, ds_part_name, "train"))
	dataset = concatenate_datasets(dataset)
	print(f"There are {len(dataset)} samples.")
	
	local_dir = args.local_dir
	hdfs_dir = args.hdfs_dir
	dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
	dataset.to_json(os.path.join(local_dir, "test.jsonl"), batch_size=len(dataset))

	if hdfs_dir is not None:
		makedirs(hdfs_dir)

		copy(src=local_dir, dst=hdfs_dir)
