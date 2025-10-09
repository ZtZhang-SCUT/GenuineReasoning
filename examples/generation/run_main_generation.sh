set -x

# data_path=/workspace/zhangzitian/code/verl/data/math500/test.parquet
data_path=/workspace/zhangzitian/code/verl/data/aime-2024.parquet
# save_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/pretrained_aime24_repeat32_temp1.0_mnt8192_gen_test.parquet
save_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/math56+code38_aime24_repeat32_temp0.0_mnt8192_gen_test.parquet
# save_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep56_math500_temp0.0_mnt8192_gen_test.parquet
# model_path=/sharedata/zhangzitian/ckpts/DAPO/DAPO-Qwen2.5-7B-Instruct-Test0811/global_step_56/actor/merged_hf_model
# model_path=/sharedata/zhangzitian/models/Qwen2.5-7B-Instruct
model_path=/sharedata/zhangzitian/ckpts/DAPO/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/global_step_38/actor/merged_hf_model

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=10240 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
