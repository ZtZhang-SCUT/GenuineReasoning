set -euo pipefail

# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep_56_math500_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/pretrained_math500_temp0.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep56_math500_temp0.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/pretrained_aime24_repeat32_temp1.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep56_aime24_repeat32_temp1.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/math56+code38_math500_temp0.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/math56+code38_aime24_repeat32_temp0.0_mnt8192_gen_test.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/step153_math_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/step153_math_temp0_mnt8192.parquet
# data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/step153_math_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/original_math_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/math500/global_step_153_math500_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/math500/global_step_140_math500_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/original_math500_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/OlympiadBench/global_step_20_olympiad_temp0_mnt8192.parquet
data_path=/tmpworkspace/zhangzitian/infer_results/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/OlympiadBench/global_step_40_olympiad_temp0_mnt8192.parquet
# python3 -m verl.trainer.main_eval \
#     data.path=$data_path \
#     data.prompt_key=prompt \
#     custom_reward_function.path=verl/trainer/ppo/reward.py \
#     custom_reward_function.name=custom_compute_score \

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
# proj_name=verl_aug_grpo
# exp_name=llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110
# dataset=olympiad_formatted

# for global_step in "${global_steps[@]}"; do
#     data_path=/tmpworkspace/zhangzitian/infer_results/$proj_name/$exp_name/math500/${global_step}_math500_temp0_mnt8192.parquet
#     echo "👉 Processing: $data_path"

#     python3 -m verl.trainer.main_eval \
#         data.path=$data_path \
#         data.prompt_key=prompt \
#         custom_reward_function.path=verl/trainer/ppo/reward.py \
#         custom_reward_function.name=custom_compute_score \

#     # 检查命令是否成功
#     if [ $? -eq 0 ]; then # [] 是条件判断，注意要有空格；$? 表示上一个命令的退出码（0=成功）
#         echo "✅ Success: $data_path"
#     else
#         echo "❌ Failed for $data_path" >&2
#     fi

#     echo "────────────────────────────────────"
# done


proj_name="verl_aug_grpo"

# # 定义三个实验名称
exp_names=(
    # "llama3_1_8b_gsm8k_ours_loop3_thres095_1108"
    # "llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110"
    "llama3_1_8b_vanilla_grpo_resplen4096_1110"
)

# 全局共享的 global_steps（如各实验 steps 不同，可改为 dict 或函数映射）
global_steps=(
    global_step_20
    global_step_40
    global_step_60
    global_step_80
    global_step_100
    global_step_120
    global_step_140
    global_step_153
)

for exp_name in "${exp_names[@]}"; do
    echo "🚀 Starting experiment: $exp_name"
    echo "────────────────────────────────────"

    for global_step in "${global_steps[@]}"; do
        save_path="/sharedata/zhangzitian/infer_results/$proj_name/$exp_name/OlympiadBench/${global_step}_olympiad_temp0_mnt8192.parquet"

        python3 -m verl.trainer.main_eval \
            data.path=$save_path \
            data.prompt_key=prompt \
            custom_reward_function.path=verl/trainer/ppo/reward.py \
            custom_reward_function.name=custom_compute_score \
        
        # 检查命令是否成功
        if [ $? -eq 0 ]; then
            echo "✅ Success: $save_path"
        else
            echo "❌ FAILED for $save_path" >&2
        fi

        echo "────────────────────────────────────"
    done

    echo "🎉 Finished experiment: $exp_name"
    echo
done

echo "✨ All experiments completed!"