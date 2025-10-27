# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>"
boxed_instruction = "{}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# add a row to each data item that represents a unique id
def make_map_fn(split, question_key, answer_key, data_source, do_extract_solution, reward_fn_extraction_type, nothink = False):

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
        # data_source = example.pop('data_source')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]


    train_dataset = train_dataset.map(function=make_map_fn("train", "question", "answer", "my_gsm8k", True, "boxed"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", "question", "answer", "my_gsm8k", True, "boxed"), with_indices=True)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
