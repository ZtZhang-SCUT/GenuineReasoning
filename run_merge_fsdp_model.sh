local_dir=/sharedata/zhangzitian/ckpts/DAPO/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/global_step_38/actor
target_dir=$local_dir/merged_hf_model
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $local_dir \
    --target_dir $target_dir \