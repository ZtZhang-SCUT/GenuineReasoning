set -x

source /workspace/zhangzitian/miniforge3/bin/activate verl050

# 启动 ray 实例
ray start --head

# wandb login
wandb login --relogin 681bad4b44acf37c1ef8cba39908a5dbd9e71d05

project_name='verl_aug_grpo'
exp_name='llama3_1_8b_vanilla_grpo_resplen4096_1110'
RAY_SHARED_DATA_HOME=/sharedata/zhangzitian
CKPTS_DIR=${CKPTS_DIR:-"${RAY_SHARED_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
validation_data_dir="${RAY_SHARED_DATA_HOME}/val_results/${project_name}/${exp_name}"
train_files='/workspace/zhangzitian/code/verl/data/gsm8k/train.parquet'
val_files='/workspace/zhangzitian/code/verl/data/gsm8k/test.parquet'

python3 -m verl.c1_trainer.main_ppo \
    exp.setting=run_vanilla_grpo_baseline \
    exp.vanilla_grpo_baseline.parsed_and_converted_data_saved_dir=/tmpworkspace/zhangzitian/data/training/verl_aug_grpo/llama3_1_8b_gsm8k_ours_loop3_thres095_1108/augment_data/parsed_and_converted_data \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/sharedata/zhangzitian/models/Meta-Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=my_gsm8k \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="disable" \
    trainer.validation_data_dir="${validation_data_dir}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=6 $@

# ray stop