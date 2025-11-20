# set -x  # 调试时显示执行的命令
set -euo pipefail

# data_path=/workspace/zhangzitian/code/verl/data/math500/test.parquet
# data_path=/workspace/zhangzitian/code/verl/data/aime-2024.parquet
data_path=/workspace/zhangzitian/code/verl/data/reason_math/test.parquet
# data_path=/workspace/zhangzitian/code/verl/data/reason_math500/test.parquet
# data_path=/workspace/zhangzitian/code/verl/data/OlympiadBench/test.parquet

# save_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/pretrained_aime24_repeat32_temp1.0_mnt8192_gen_test.parquet
# save_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/math56+code38_aime24_repeat32_temp0.0_mnt8192_gen_test.parquet
# save_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep56_math500_temp0.0_mnt8192_gen_test.parquet
save_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/step153_math_temp0_mnt8192.parquet
# save_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/step153_math_temp0_mnt8192.parquet
# save_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/step153_math_temp0_mnt8192.parquet
# save_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/step153_olympiad_temp0_mnt8192.parquet
# save_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/original_math500_temp0_mnt8192.parquet

# model_path=/sharedata/zhangzitian/ckpts/DAPO/DAPO-Qwen2.5-7B-Instruct-Test0811/global_step_56/actor/merged_hf_model
# model_path=/sharedata/zhangzitian/models/Qwen2.5-7B-Instruct
# model_path=/sharedata/zhangzitian/ckpts/DAPO/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/global_step_38/actor/merged_hf_model
model_path=/sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/global_step_153/actor/merged_hf_model
# model_path=/sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_153/actor/merged_hf_model
# model_path=/sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_153/actor/merged_hf_model
# model_path=/sharedata/zhangzitian/models/Meta-Llama-3.1-8B-Instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=1024 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=10240 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9

python3 -m verl.trainer.main_eval \
    data.path=$save_path \
    data.prompt_key=prompt \
    custom_reward_function.path=verl/trainer/ppo/reward.py \
    custom_reward_function.name=custom_compute_score \

# 数组遍历
# proj_name=verl_aug_grpo
# exp_name=llama3_1_8b_gsm8k_ours_loop3_thres095_1108
# global_steps=(
#     global_step_20
#     global_step_40
#     global_step_60
#     global_step_80
#     global_step_100
#     global_step_120
#     global_step_140
#     global_step_153
# )
# data_path=/workspace/zhangzitian/code/verl/data/reason_math500/test.parquet
# for global_step in "${global_steps[@]}"; do
#     model_path=/sharedata/zhangzitian/ckpts/$proj_name/$exp_name/$global_step/actor/merged_hf_model
#     save_path=/tmpworkspace/zhangzitian/infer_results/$proj_name/$exp_name/math500/${global_step}_math500_temp0_mnt8192.parquet
#     echo "👉 Processing: $model_path"
#     python3 -m verl.trainer.main_generation \
#         trainer.nnodes=1 \
#         trainer.n_gpus_per_node=1 \
#         data.path=$data_path \
#         data.prompt_key=prompt \
#         data.n_samples=1 \
#         data.batch_size=1024 \
#         data.output_path=$save_path \
#         model.path=$model_path \
#         +model.trust_remote_code=True \
#         rollout.temperature=0.0 \
#         rollout.prompt_length=1024 \
#         rollout.response_length=8192 \
#         rollout.max_num_batched_tokens=10240 \
#         rollout.tensor_model_parallel_size=1 \
#         rollout.gpu_memory_utilization=0.9
        
#     # 检查命令是否成功
#     if [ $? -eq 0 ]; then # [] 是条件判断，注意要有空格；$? 表示上一个命令的退出码（0=成功）
#         echo "✅ Success: $save_path"
#     else
#         echo "❌ Failed for $model_path" >&2
#     fi
#     echo "────────────────────────────────────"
# done


# 字典遍历
#!/bin/bash

# proj_name="verl_aug_grpo"

# # 定义实验名称
# exp_names=(
#     "llama3_1_8b_gsm8k_ours_loop3_thres095_1108"
#     "llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110"
# )

# # 全局共享的 global_steps（如各实验 steps 不同，可改为 dict 或函数映射）
# global_steps=(
#     global_step_20
#     global_step_40
#     global_step_60
#     global_step_80
#     global_step_100
#     global_step_120
#     global_step_140
#     global_step_153
# )

# declare -A exp_global_steps
# steps_exp1=(
#     global_step_20
#     global_step_40
#     global_step_60
#     global_step_80
#     global_step_100
#     global_step_120
#     global_step_140
#     global_step_153
# )
# steps_exp2=(
#     global_step_20
#     global_step_40
#     global_step_60
#     global_step_80
#     global_step_100
#     global_step_120
#     global_step_140
#     global_step_153
# )
# steps_exp3=(
#     global_step_20
#     global_step_40
#     global_step_60
#     global_step_80
#     global_step_100
#     global_step_120
#     global_step_140
#     global_step_153
# )

# exp_global_steps["llama3_1_8b_gsm8k_ours_loop3_thres095_1108"]="steps_exp1"

# exp_global_steps["llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110"]="steps_exp2"
# exp_global_steps["llama3_1_8b_vanilla_grpo_resplen4096_1110"]="steps_exp3"

# data_path="/workspace/zhangzitian/code/verl/data/reason_math500/test.parquet"
# data_path="/workspace/zhangzitian/code/verl/data/OlympiadBench/test.parquet"
# data_path="/workspace/zhangzitian/code/verl/data/reason_math/test.parquet"

# for exp_name in "${!exp_global_steps[@]}"; do
#     echo "🚀 Starting experiment: $exp_name"
#     echo "👉 Evaluating: $data_path"
#     echo "────────────────────────────────────"
#     steps_var=("${exp_global_steps[$exp_name]}")
#     eval "steps=(\"\${${steps_var}[@]}\")"

#     for global_step in "${steps[@]}"; do
#         model_path="/sharedata/zhangzitian/ckpts/$proj_name/$exp_name/$global_step/actor/merged_hf_model"
#         save_path="/tmpworkspace/zhangzitian/infer_results/$proj_name/$exp_name/MATH/${global_step}_math_temp0_mnt8192.parquet"
#         share_data_path="/sharedata/zhangzitian/infer_results/$proj_name/$exp_name/MATH/${global_step}_math_temp0_mnt8192.parquet"

#         echo "👉 Processing: $model_path"
#         echo "create folder: $(dirname "$share_data_path")"

#         # 创建 save_path 的父目录（避免 parquet 写入失败）
#         mkdir -p "$(dirname "$save_path")"
#         mkdir -p "$(dirname "$share_data_path")"
#         ls -la "$(dirname "$share_data_path")"

#         # 执行推理
#         python3 -m verl.trainer.main_generation \
#             trainer.nnodes=1 \
#             trainer.n_gpus_per_node=1 \
#             data.path="$data_path" \
#             data.prompt_key=prompt \
#             data.n_samples=1 \
#             data.batch_size=1024 \
#             data.output_path="$save_path" \
#             model.path="$model_path" \
#             +model.trust_remote_code=True \
#             rollout.temperature=0.0 \
#             rollout.prompt_length=1024 \
#             rollout.response_length=8192 \
#             rollout.max_num_batched_tokens=10240 \
#             rollout.tensor_model_parallel_size=1 \
#             rollout.gpu_memory_utilization=0.9
        
#         python3 -m verl.trainer.main_eval \
#             data.path=$save_path \
#             data.prompt_key=prompt \
#             custom_reward_function.path=verl/trainer/ppo/reward.py \
#             custom_reward_function.name=custom_compute_score \
        
#         # 检查命令是否成功
#         if [ $? -eq 0 ]; then
#             echo "✅ Success: $save_path"
#             # cp $save_path $share_data_path
#         else
#             echo "❌ FAILED for $model_path" >&2
#         fi

#         echo "────────────────────────────────────"
#     done

#     echo "🎉 Finished experiment: $exp_name"
#     echo
# done

# echo "✨ All experiments completed!"