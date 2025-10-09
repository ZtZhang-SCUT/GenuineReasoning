set -x

# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep_56_math500_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/pretrained_math500_temp0.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep56_math500_temp0.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/pretrained_aime24_repeat32_temp1.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Test0811/gstep56_aime24_repeat32_temp1.0_mnt8192_gen_test.parquet
# data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/math56+code38_math500_temp0.0_mnt8192_gen_test.parquet
data_path=/home/aigc/DAPO-Qwen2.5-7B-Instruct-Coder1-12k-0817/math56+code38_aime24_repeat32_temp0.0_mnt8192_gen_test.parquet

python3 -m verl.trainer.main_eval \
    data.path=$data_path \
    data.prompt_key=prompt \
    custom_reward_function.path=verl/trainer/ppo/reward.py \
    custom_reward_function.name=custom_compute_score \