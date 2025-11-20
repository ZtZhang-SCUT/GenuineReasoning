# local_dir=/sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/global_step_140/actor
# target_dir=$local_dir/merged_hf_model
# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir $local_dir \
#     --target_dir $target_dir \


#!/bin/bash

# 定义所有 local_dir 的列表（可换行提高可读性）
local_dirs=(
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_20/actor
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_40/actor
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_60/actor
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_80/actor
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_100/actor
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_120/actor
    # /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_vanilla_grpo_resplen4096_1110/global_step_140/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_20/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_40/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_60/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_80/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_100/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_120/actor
    /sharedata/zhangzitian/ckpts/verl_aug_grpo/llama3_1_8b_aug_gsm8k_vanilla_grpo_resplen4096_1110/global_step_140/actor
)

# 遍历每个 local_dir
for local_dir in "${local_dirs[@]}"; do
    echo "👉 Processing: $local_dir"
    
    # 检查目录是否存在
    if [ ! -d "$local_dir" ]; then
        echo "❌ Directory not found: $local_dir" >&2
        continue
    fi

    target_dir="$local_dir/merged_hf_model"
    
    # 可选：跳过已存在的 merged_hf_model（避免重复）
    if [ -d "$target_dir" ]; then
        echo "⚠️  Skipped: merged_hf_model already exists at $target_dir"
        continue
    fi

    # 执行合并命令
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --target_dir "$target_dir"

    # 检查命令是否成功
    if [ $? -eq 0 ]; then
        echo "✅ Success: $target_dir"
    else
        echo "❌ Failed for $local_dir" >&2
    fi

    echo "────────────────────────────────────"
done