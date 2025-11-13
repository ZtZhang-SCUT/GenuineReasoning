# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]

from verl.utils.multi_scene_data.process_data import make_map_fn, ModifiedGaussian
from verl.utils.fs import fast_safe_copy_to_sharedata, safe_copy_to_sharedata
from numpy import random
from typing import Any, Dict, Union, Optional, Callable
from datetime import datetime
# from pathlib import Path
from glob import glob
import re

def wrong_rate2sample_probs(map_func: Callable, xs: np.ndarray) -> np.ndarray:
    """
    将错误率numpy数组转换为采样概率分布（并行化计算）
    :param xs: 错误率numpy数组（元素范围[0,1]）
    :return: 与输入同形状的概率数组（非零元素和为1）
    """
    if not isinstance(xs, np.ndarray):
        xs = np.array(xs)
    if xs.ndim != 1:
        raise ValueError("输入必须是一维numpy数组")
    
    # 向量化计算所有x的函数值（无循环，并行效率高）
    fxs = map_func(xs)  # 直接传入数组，内部用向量化操作
    
    # 过滤非零元素（用向量化掩码，替代列表推导）
    non_zero_mask = fxs > 1e-9  # 布尔数组，形状与xs一致
    non_zero_values = fxs[non_zero_mask]  # 提取非零值
    
    # 处理全零情况
    if non_zero_values.size == 0:
        return np.zeros_like(xs, dtype=np.float64)
    
    # 向量化归一化（避免循环，直接用掩码赋值）
    total = non_zero_values.sum()
    normalized = np.zeros_like(fxs, dtype=np.float64)
    normalized[non_zero_mask] = non_zero_values / total  # 仅对非零位置赋值
    
    return normalized

def _make_json_serializable(val: Any):
    """Convert common types to JSON-serializable equivalents, preserving dict/list structure."""
    if isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, np.ndarray):
        if val.ndim == 0:
            return val.item()
        elif val.ndim == 1:
            return [_make_json_serializable(v) for v in val]
        else:
            return None  # skip multi-dim arrays
    elif isinstance(val, dict):
        return {k: _make_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_make_json_serializable(v) for v in val]
    elif isinstance(val, (str, int, float, bool)) or val is None:
        return val
    else:
        return str(val)  # fallback: convert to string



class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        if self.config.get("meta_trainer", None).get("scene_generator", None):
            from verl.utils.multi_scene_data.scene_generator import SceneGenerator
            from verl.utils.multi_scene_data.async_scene_generator_v5 import SceneGeneratorManager

            # self.scene_generator = SceneGenerator(
            #     api_url=self.config.meta_trainer.scene_generator.api_url,
            #     api_key=self.config.meta_trainer.scene_generator.api_key,
            #     model_name=self.config.meta_trainer.scene_generator.model_name,
            #     timeout=180
            #     # scenes=self.config.scene_generator.get("scenes", ["地理", "医学", "金融", "农业", "日常生活", "税务"]),
            # )

            self.scene_generator = SceneGeneratorManager(
                model_name=self.config.meta_trainer.scene_generator.model_name,
                api_urls=[self.config.meta_trainer.scene_generator.api_url],
                api_keys=[self.config.meta_trainer.scene_generator.api_key],
                timeout=180
            )
        else:
            self.scene_generator = None

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        ) # 确保分布式训练中数据能均匀分配到各 GPU，避免批次处理异常，维持训练稳定性与算法有效性。若不能整除，某些 GPU 会空闲或报错，导致训练崩溃或效率低下

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        # 由于aws存储系统的特殊性，这里需要先删再写
        if os.path.exists(filename):  # remove the old one
            print(f"Removing old val_result saved path: {filename}")
            os.remove(filename)

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"  # {'val-aux/my_gsm8k/reward/mean@1': 0.6876421531463229, 'val-aux/my_gsm8k/score/mean@1': 0.6876421531463229, 'val-core/my_gsm8k/acc/mean@1': 0.8438210765731615}
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items(): # 这里没看懂
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )

        if os.path.exists(local_latest_checkpointed_iteration):  # remove the old one
            print(f"Removing old latest checkpointed iteration file: {local_latest_checkpointed_iteration}")
            os.remove(local_latest_checkpointed_iteration)
        
        with open(local_latest_checkpointed_iteration, "w") as f:  # atomic write
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _load_generate_history(self):
        """
        恢复历史生成状态
        要加载history_questions2variants
        修改self.total_training_steps
        """
        history_question2variants = defaultdict(list)
        if self.config.meta_trainer.resume_mode == "disable":
            return 0

        history_folder = self.config.meta_trainer.default_history_dir
        if not os.path.isabs(history_folder):
            working_dir = os.getcwd()
            history_folder = os.path.join(working_dir, history_folder)
        
        # find last global_step，直接从 self.global_steps 开始就行了
        # latest_step = find_latest_step(history_folder)

        global_step_path = find_latest_history_path(history_folder, self.global_steps)  # None if no latest

        if self.config.meta_trainer.resume_mode == "auto":
            if global_step_path is None:
                print("Can't found corresponding history generation path, training from scratch")
                return 0
        else:
            if self.config.resume_mode == "resume_path":
                assert isinstance(self.config.meta_trainer.resume_from_path, str)   # 直接指定json文件
                global_step_path = self.config.meta_trainer.resume_from_path
                if not os.path.isabs(global_step_path):
                    working_dir = os.getcwd()
                    global_step_path = os.path.join(working_dir, global_step_path)
        
        print(f"Load from history_question2variants path: {global_step_path}")
        
    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _evaluate_current_batch(self, batch: DataProto, repeat_times: int=1, metric_name="acc") -> tuple[DataProto, dict]:
        """确认当前batch中哪些问题是模型会答错的"""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        # batch = DataProto.from_single_dict(batch_dict)

        # repeat test batch
        # TODO: 涉及到具体怎么衡量模型答错：是多次尝试都答错 还是 有一次答错就算答错，换言之，pass@k的k取多少，暂时取1
        # assert self.config.meta_trainer.val_kwargs.n == 1, f"for now, just support n=1, but got {self.config.meta_trainer.val_kwargs.n=}"
        # assert repeat_times == 1, f"for now, just support n=1, but got {self.config.meta_trainer.val_kwargs.n=}"
        # 要求每个item都有uid这个属性
        batch = batch.repeat(
            repeat_times=repeat_times, interleave=True
        )

        # we only do validation on rule-based rm
        if self.config.reward_model.enable and batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            return batch, {}  # TODO: 返回什么后面确认

        # Store original inputs
        input_ids = batch.batch["input_ids"]
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)

        gen_batch = self._get_gen_batch(batch)
        # TODO: repeat_times = 1 时 do_sample=False
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            # "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            # "validate": True,
            "global_steps": self.global_steps,
        }
        print(f"test_gen_batch meta info: {gen_batch.meta_info}")

        # pad to be divisible by dp_size
        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        if not self.async_rollout_mode:
            output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        else:
            output_gen_batch_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)

        # unpad
        output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
        print("validation generation end")

        # Store generated outputs
        output_ids = output_gen_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        batch = batch.union(output_gen_batch)
        # batch.meta_info["validate"] = True

        # evaluate using reward_function
        result = self.val_reward_fn(batch, return_dict=True)
        reward_tensor = result["reward_tensor"]

        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)
        # test_batch.batch["token_level_scores"] = reward_tensor  # 后面重新计算
        reward_extra_infos_dict["reward"].extend(scores)  # 根据reward字段可以判断哪些题答对，哪些题答错，但是要注意最好是0,1奖励或1，-1奖励
        
        print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
        if "reward_extra_info" in result:
            for key, lst in result["reward_extra_info"].items():
                reward_extra_infos_dict[key].extend(lst)
                print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

        if reward_extra_infos_dict:
            batch.non_tensor_batch.update(
                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
            )

        # collect num_turns of each prompt
        if "__num_turns__" in batch.non_tensor_batch:
            sample_turns.append(batch.non_tensor_batch["__num_turns__"])

        data_source_lst.append(batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 如果 metric_name 是 acc, 说明只有自己定义的准确性奖励（格式奖励）作为筛选标准
        # 如果 metric_name 是 seq_final_reward, 说明是准确性奖励（格式奖励） + 自定义奖励（如长度等）- kl 惩罚，即所有奖励都参与筛选
        # 如果 metric_name 是 seq_reward, 说明是准确性奖励（格式奖励）+ 自定义奖励（如长度等）作为筛选标准
        if metric_name == "seq_final_reward":    
            # Turn to numpy for easier filtering
            batch.non_tensor_batch["seq_final_reward"] = (
                batch.batch["token_level_rewards"].sum(dim=-1).numpy()
            )
        elif metric_name == "seq_reward":
            batch.non_tensor_batch["seq_reward"] = (
                batch.batch["token_level_scores"].sum(dim=-1).numpy()
            )

        # Collect the sequence reward for each trajectory
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
        ):
            prompt_uid2metric_vals[uid].append(metric_val)

        prompt_uid2metric_mean = {}
        for prompt_uid, metrics_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_mean[prompt_uid] = np.mean(metrics_vals)
            
        return batch, prompt_uid2metric_mean
    
    def _generate_augdata_until_target_size_sync(self, tobe_aug_batch: DataProto, target_size: int, history_question2variants: dict[str, list]={}, save_file:str="") -> list[dict]:
        """Use external scene generator to generate multi-scene data.

        Args:
            raw_prompts (list[str]): List of raw prompts.
            target_size
        Returns:
            list[dict]: List of generated scene data.
        """
        aug_data_list = []

        if self.scene_generator is None:
            raise ValueError("scene_generator is required for meta-training but not initialized.")

        attempts = 0
        max_attempts = target_size * 2  # 防止死循环
        while len(aug_data_list) < target_size and attempts < max_attempts:
            # question2variants = defaultdict(list)

            for data_item in tobe_aug_batch:
                if len(aug_data_list) >= target_size:
                    break
                attempts += 1
                original_question_key = "problem"
                original_question = data_item.non_tensor_batch[original_question_key]

                # 生成场景变体
                try:
                    if original_question not in history_question2variants:
                        history_question2variants[original_question] = []
                    print(f"[Debug] Attempt: {attempts}, Generating for: {original_question[:100]}")
                    variant = self.scene_generator.generate_and_save(
                            original_question=original_question,
                            previous_aug_questions=history_question2variants[original_question],
                            save_file=save_file
                        )
                    if variant:
                        # TODO: 可加入有效性检查（如 answer 是否一致）
                        question_key = "question"
                        history_question2variants[original_question].append(variant[question_key])
                        aug_data_list.append({
                            "question": variant[question_key],
                            "solution": variant["solution"],
                            "data_source": f"{self.config.meta_trainer.scene_generator.model_name}_generated",
                        })

                except Exception as e:
                    print(f"[Warning] Scene generation failed for idx {data_item.non_tensor_batch[original_question_key]}: {e}")
                    continue
        
        return aug_data_list, history_question2variants
    
    def _generate_augdata_until_target_size(self, tobe_aug_batch: DataProto, target_size: int, history_question2variants: dict[str, list]=defaultdict[list], save_file: str="") -> tuple[list[dict], dict]:
        """Use external scene generator to generate multi-scene data in batches.

        Args:
            tobe_aug_batch (DataProto): Batch of prompts to be augmented.
            target_size (int): Target number of augmented samples.
            history_question2variants (dict): History of generated variants for each original question.
            save_file (str): File to save raw outputs.
        Returns:
            tuple[list[dict], dict]: (aug_data_list, updated_history_question2variants)
        """
        aug_data_list = []
        remaining_target = target_size

        if self.scene_generator is None:
            raise ValueError("scene_generator_manager is required for meta-training but not initialized.")

        attempts = 0
        max_attempts = target_size * 2  # 防止死循环
        
        # 将 tobe_aug_batch 转换为可迭代的列表
        batch_items = list(tobe_aug_batch)  # 假设 tobe_aug_batch 可迭代
        
        while len(aug_data_list) < target_size and attempts < max_attempts:
            # 计算当前批次需要生成的数量
            current_batch_size = min(remaining_target, len(batch_items))
            
            # 构建批量请求数据
            batch_requests = []
            for i in range(current_batch_size):
                data_item = batch_items[i + attempts % len(batch_items)]  # 循环使用batch中的数据
                original_question_key = "problem"
                original_question = data_item.non_tensor_batch[original_question_key]
                
                # 获取历史变体
                previous_aug_questions = history_question2variants.get(original_question, [])
                
                batch_requests.append({
                    "original_question": original_question,
                    "previous_aug_questions": previous_aug_questions,
                    # "save_file": save_file,
                    # "is_save": True
                })
            
            try:
                # 批量生成
                print(f"[Debug] Attempt: {attempts}, Batch size: {len(batch_requests)}, Target remaining: {remaining_target}")
                
                batch_results = self.scene_generator.generate_batch(batch_requests, save_file=save_file, is_save=True)
                
                # 处理批量结果
                for i, (request_data, variant) in enumerate(zip(batch_requests, batch_results)):
                    if len(aug_data_list) >= target_size:
                        break
                        
                    original_question = request_data["original_question"]
                    
                    if variant:  # 如果生成成功
                        question_key = "question"
                        # 更新历史记录
                        history_question2variants[original_question].append(variant[question_key])
                        
                        # 添加到结果列表
                        aug_data_list.append({
                            "question": variant[question_key],
                            "solution": variant["solution"],
                            "data_source": f"{self.config.meta_trainer.scene_generator.model_name}_generated",
                        })
                        
                        print(f"[Debug] Generated variant {len(aug_data_list)}/{target_size}: {variant[question_key][:50]}...")
                    
                    attempts += 1
                remaining_target = target_size - len(aug_data_list)
            
            except Exception as e:
                print(f"[Warning] Batch generation failed: {e}")
                pass
        print(f"[Debug] Final: Generated {len(aug_data_list)} samples, Target: {target_size}")
        return aug_data_list, history_question2variants

    def _convert_to_experted_data_format(self, data: list[dict]) -> list[dict]:
        map_fn = make_map_fn(split="train", question_key="question", answer_key="solution", do_extract_solution=True, reward_fn_extraction_type="boxed")
        formatted_data = []
        for i, d in enumerate(data):
            formatted_data.append(map_fn(d, i))
        return formatted_data
    
    def _save_data_to_jsonl_and_parquet(self, data: list[dict], save_dir: str, prefix_name: str="generated_scene_data") -> str:
        """Save data to json and parquet files.

        Args:
            data (list[dict]): List of data entries.
            save_path (str): Path to save the files.
        Returns:
            str: Path to the saved parquet file.
        """
        import pandas as pd

        step = self.global_steps

        os.makedirs(save_dir, exist_ok=True)
        jsonl_path = os.path.join(save_dir, f"{prefix_name}_num_samples_{len(data)}_global_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        parquet_path = os.path.join(save_dir, f"{prefix_name}_num_samples_{len(data)}_global_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")

        # save to json
        with open(jsonl_path, "w", encoding="utf-8") as f:
            # json.dump(data, f, ensure_ascii=False, indent=2)
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[Info] Saved generated scene data to {jsonl_path}")

        # save to parquet
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path)
        print(f"[Info] Saved generated scene data to {parquet_path}")

        return parquet_path


    def _create_temp_dataloader_from_path(self, generated_data_path: str, batch_size: int=None):
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from torch.utils.data import SequentialSampler, DataLoader

        # create dataset
        dataset = RLHFDataset(
            data_files=[generated_data_path],
            config=self.config.data, 
            tokenizer=self.tokenizer, 
            processor=self.processor
        )
        print(f"Created temp dataset with {len(dataset)} samples.")

        # create sampler
        sampler = SequentialSampler(data_source=dataset)  # iterate through the dataset in order

        # create dataloader
        num_workers = self.config.data["dataloader_num_workers"]
        dataloader = DataLoader(
            dataset=dataset,
            # batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            batch_size=batch_size if batch_size else len(dataset),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=default_collate_fn,
            sampler=sampler,
        )
        print(f"Created temp dataloader with {len(dataloader)} batches.")
        return dataloader
    
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    def _compute_batch(self, batch: DataProto, metrics: dict, timing_raw: dict, skip_gen: bool=True) -> tuple[DataProto, dict]:
        
        if not skip_gen:
            # pop keys for generation
            gen_batch = self._get_gen_batch(batch)

            # pass global_steps to trace
            # gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)
        
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                raise NotImplementedError("Not support REMAX adv_estimator yet.")
                with marked_timer("gen_max", timing_raw, color="purple"):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    if not self.async_rollout_mode:
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                    else:
                        gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # batch.non_tensor_batch["uid"] = np.array(
            #     [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            # )

            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        # TODO: Decouple the DP balancing and mini-batching.
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        with marked_timer("reward", timing_raw, color="green"):
            # compute reward model score
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)
            
            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            
        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                # TODO: we may want to add diff of probs too.
                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                actor_old_log_probs = batch.batch["old_log_probs"]
                attention_mask = batch.batch["attention_mask"]
                responses = batch.batch["responses"]
                response_length = responses.size(1)
                response_mask = attention_mask[:, -response_length:]

                rollout_probs = torch.exp(rollout_old_log_probs)
                actor_probs = torch.exp(actor_old_log_probs)
                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                metrics.update(
                    {
                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    }
                )
                
        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch) # [bs, seq_len] 每个 token (state) 都有一个不同的 value
                batch = batch.union(values)                

        with marked_timer("adv", timing_raw, color="brown"):
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict: # 好像有点多余了
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # compute advantages, executed on the driver process
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )
        
        return batch, metrics, reward_extra_infos_dict
    
    def _standard_ppo_step(self, batch: DataProto, metrics: dict, timing_raw: dict, skip_gen: bool=True):
        batch, metrics, reward_extra_infos_dict = self._compute_batch(batch, metrics, timing_raw, skip_gen)

        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        
        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )
                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )
        
        return metrics
        
    def _get_tobe_augment_prompt_uids(self, uid2metric: dict, n_to_augment: int) -> list[str]:
        """
        根据每个prompt的acc，选出需要被增强的uid，最终得到大于等于n_to_augment个待增强的prompt
        """
        # # 把回答准确率在0~0.8之间的prompt拿去生成新问题（排除已完美掌握或完全无效的样本）52
        tobe_augmented_prompt_uids = [
            uid
            for uid, estimated_acc in uid2metric.items()
            if 0 <= estimated_acc <= 0.8
        ]
        if len(tobe_augmented_prompt_uids) >= n_to_augment:
            filted_kept_ori_order_prompt_uids = tobe_augmented_prompt_uids
        else:
            # 对 metric_mean 进行升序排序，截取n_to_augment个prompt去增强（准确率越低，难度越高，越优先增强）
            sorted_prompt_uids_by_metric = sorted(  # list[tuple]
                uid2metric.items(),
                key=lambda x: x[1]
            )
            worst_uids = [uid for uid, _ in sorted_prompt_uids_by_metric[:n_to_augment]]
            filted_kept_ori_order_prompt_uids = [uid for uid in list(uid2metric.keys()) if uid in worst_uids]
        return filted_kept_ori_order_prompt_uids
    

    def _save_dataproto_to_jsonl(
        self,
        dataproto: "DataProto",
        filepath: str,
        include_meta_info: bool = True,
        one_id_saved_once: bool = True
    ):
        n_samples = len(dataproto)
        if n_samples == 0:
            print(f"[Warning] DataProto is empty.")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        already_saved_uid = []
        sample_cnt = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for i in range(n_samples):
                # 一个id只保存一次
                if one_id_saved_once:
                    uid = dataproto.non_tensor_batch["uid"][i]
                    if uid in already_saved_uid:
                        continue
                    already_saved_uid.append(uid)

                sample = {}

                # Add non_tensor_batch fields (preserving nested structure)
                for key, arr in dataproto.non_tensor_batch.items():
                    val = arr[i]
                    serial_val = _make_json_serializable(val)
                    if serial_val is not None:
                        sample[key] = serial_val

                # Optionally add meta_info as top-level fields
                if include_meta_info and dataproto.meta_info:
                    serial_meta = _make_json_serializable(dataproto.meta_info)
                    # if isinstance(serial_meta, dict):
                    #     sample.update(serial_meta)
                    # else:
                    #     sample["meta_info"] = serial_meta
                    sample["meta_info"] = serial_meta

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sample_cnt += 1

        print(f"Saved {sample_cnt} samples to {filepath}")
    
    def _save_dict_to_jsonl(
        self,
        data: Union[Dict[str, Any], list],
        filepath: str,
        *,   # 之后的所有参数必须以关键字（keyword-only arguments）的形式传入，不能通过位置传参。
        mode: str = "w",
        encoding: str = "utf-8",
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        indent: Optional[int] = None,
    ) -> None:
        """
        将字典或字典列表安全地保存为 JSONL 文件。

        参数:
            data: 单个 dict 或 dict 列表。若为单个 dict，则写入一行；若为 list，则每项一行。
            filepath: 输出文件路径。
            mode: 文件打开模式，默认 "w"（覆盖），可用 "a" 追加。
            encoding: 文件编码，默认 "utf-8"。
            ensure_ascii: 是否转义非 ASCII 字符，默认 False（保留中文等）。
            sort_keys: 是否对字典键排序（用于确定性输出）。
            indent: 一般 JSONL 不缩进，保留为 None。

        注意:
            - 自动处理嵌套结构、datetime、set、numpy 类型等。
            - 不会抛出 json.JSONEncodeError。
        """
        # 确保父目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 统一处理为列表
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise ValueError("If data is a list, all items must be dictionaries.")
            records = data
        else:
            raise TypeError("data must be a dict or a list of dicts.")

        # 预处理每条记录
        processed_records = []
        for record in records:
            try:
                clean_record = _make_json_serializable(record)
                processed_records.append(clean_record)
            except Exception as e:
                # 理论上 _robust_serialize 不应抛出异常，但防御性处理
                raise RuntimeError(f"Failed to preprocess record: {record}") from e

        # 写入 JSONL
        with open(filepath, mode, encoding=encoding) as f:
            for record in processed_records:
                line = json.dumps(
                    record,
                    ensure_ascii=ensure_ascii,
                    sort_keys=sort_keys,
                    indent=indent,
                )
                f.write(line + "\n")

    def _save_dict_to_json(
        self,
        data: Union[Dict[str, Any], list],
        filepath: str,
        *,   # 之后的所有参数必须以关键字（keyword-only arguments）的形式传入，不能通过位置传参。
        encoding: str = "utf-8",
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        indent: Optional[int] = 2,
    ) -> None:
        """
        将字典或字典列表安全地保存为 JSONL 文件。

        参数:
            data: 单个 dict 或 dict 列表。若为单个 dict，则写入一行；若为 list，则每项一行。
            filepath: 输出文件路径。
            mode: 文件打开模式，默认 "w"（覆盖），可用 "a" 追加。
            encoding: 文件编码，默认 "utf-8"。
            ensure_ascii: 是否转义非 ASCII 字符，默认 False（保留中文等）。
            sort_keys: 是否对字典键排序（用于确定性输出）。
            indent: 一般 JSONL 不缩进，保留为 None。

        注意:
            - 自动处理嵌套结构、datetime、set、numpy 类型等。
            - 不会抛出 json.JSONEncodeError。
        """
        # 确保父目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 验证并预处理数据
        if isinstance(data, dict):
            clean_data = _make_json_serializable(data)
        elif isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise ValueError("If data is a list, all items must be dictionaries.")
            clean_data = _make_json_serializable(data)
        else:
            raise TypeError("data must be a dict or a list of dicts.")

        # 写入标准 JSON 文件
        with open(filepath, "w", encoding=encoding) as f:
            json.dump(
                clean_data,
                f,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                indent=indent,
            )

    def _is_early_exit(self, metric: float, threshold: float, mode: str="bigger") -> bool:
        if mode == "bigger":
            if metric >= threshold:
                return True
            return False
        elif mode == "smaller":
            if metric <= threshold:
                return True
            return False
        else:
            raise ValueError(f"Unknown compare mode, expected 'bigger' or 'smaller', but got '{mode}'")

    def _restore_training_state_from_history(self, total_training_steps, current_global_step):
        """
        自动恢复：
        1. history_question2variants 字典；
        2. 根据 augment_history 文件推断 total_training_steps。

        依赖文件命名规则：
        global_training_step_{step}_outer_loop_{outer_loop}_augment_history_*.jsonl
        """
        base_dir = f"{self.config.meta_trainer.base_sharedata_data_dir}/" \
               f"{self.config.trainer.project_name}/" \
               f"{self.config.trainer.experiment_name}/augment_history"
        
        if not os.path.exists(base_dir):
            print(f"[Restore] augment_history directory not found: {base_dir}")
            return defaultdict(list), total_training_steps

        # 1. 匹配文件名中的 step 与 outer_loop
        pattern = re.compile(r"global_training_step_(\d+)_outer_loop_(\d+)_augment_history_.*\.jsonl$")
        records = []
        for file_path in glob(os.path.join(base_dir, "*.jsonl")):
            m = pattern.search(os.path.basename(file_path))
            if m:
                step, outer_loop = int(m.group(1)), int(m.group(2))
                if step <= current_global_step:  # 过滤掉 record[0] > current_global_step 的记录
                    records.append((step, outer_loop, file_path))

        if not records:
            print(f"[Restore] No augment_history files found in {base_dir}")
            return defaultdict(list), total_training_steps

        # 2. 按 step, outer_loop 排序
        records.sort(key=lambda x: (x[0], x[1]))

        # 3. 找到当前 step 对应的 history_file
        history_file = [r[2] for r in records if r[0] == current_global_step][0]
        print(f"[Restore] Latest augment history file: {history_file}")
        print(f"[Restore] Latest global_training_step={current_global_step}")

        # 4. 加载最后一个文件中的 history_question2variants
        history_question2variants = defaultdict(list)
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for q, v in data.items():
                    history_question2variants[q].extend(v)
        except Exception as e:
            print(f"[Restore] Failed to load {history_file}: {e}")

        # 5. 动态调整 total_training_steps
        # 注意：self.global_steps 已由 _load_checkpoint() 设置为上次 ckpt 对应 step。
        #      这里我们根据 augment_history 推断上一个 step 的 outer_loop 是否提早结束。
        repeat_count = self.config.meta_trainer.get("outer_loop_repeat", 1)
        to_counter = [record[1] for record in records]

        # 假如最后的 outer_loop 不是 repeat_count-1，那么要找到最近的0的前一个，already_escape 重新计算，同时获取 current_outer_loop=最后的outer_loop
        next_outer_loop = -1
        if to_counter[-1] != repeat_count-1:  # 说明最后一轮是不完整的
            next_outer_loop = to_counter[-1] + 1
            last_0_index = -1 - to_counter[::-1].index(0)
            to_counter = to_counter[:last_0_index]  # 前面的都是完整的
        
        counter = Counter(to_counter)
        print(f"counter: {counter}, keys: {counter.keys()}")

        reduced_steps = counter[0]*repeat_count - sum(list(counter.values()))
        old_total = total_training_steps
        total_training_steps -= reduced_steps
        print(f"[Restore] Adjust total_training_steps: {old_total} -> {total_training_steps} "
            f"(reduced {reduced_steps} due to early exit). "
            f"The outer_loop_index of step {current_global_step+1} should be {next_outer_loop}")
        
        return history_question2variants, total_training_steps, next_outer_loop

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        """
        for epoch in range(total_epochs):
            for ori_batch in train_loader:
                for t = 0, ..., T repeat iterations:
                    在ori_batch上rollout(每个样本k次)，估计每个样本的准确率
                    从ori_batch中选出 至少2/3*train_batch_size（为了覆盖各种难度的样本）
                        - 筛选准确率从0~0.8的样本进行增强
                        - 如果数量不足2/3*train_batch_size，根据准确率降序排序，再截取2/3*train_batch_size个进行改写（准确率低的优先增强）
                        - 如果超过2/3*train_batch_size，则保留
                    针对这些样本生成跨场景变体（每个iterations生成的变体都不一样），直到aug_batch达到目标大小(设为train_batch_size)
                    aug_batch上rollout，统计每个样本的准确率
                    拼接 ori_batch 和 aug_batch 成 merged_batch，基于每个样本的准确率得到每个样本的采样概率
                        - 过易的题目采样概率为0，难度适中偏难的样本概率较高，过难样本样本概率较低
                    基于每个样本的采样概率，从 merged_batch 中采出 train_batch_size 个样本构成最终的 train_batch
                    在train_batch上进行强化学习训练
                    早停策略（模型在当前ori_batch+aug_batch上的准确率达到阈值后退出最内层迭代，转向处理下一个batch）
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0   # global_step 通常被定义为截止目前模型参数更新的总次数

        # load checkpoint before doing anything
        # self._load_checkpoint()
        self.global_steps = 140

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.total_training_steps *= self.config.meta_trainer.get("outer_loop_repeat", 1)  # 最大训练步数
        # TODO: 从路径中恢复 history_question2variants
        history_question2variants = defaultdict(list) # 记录每个问题的历史增强记录（增量式）
        history_question2variants, self.total_training_steps, next_outer_loop_idx = self._restore_training_state_from_history(self.total_training_steps, self.global_steps)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0  # 完成一个step的最大耗时记录

        for epoch in range(self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                # print(f"[Debug] key of batch_dict: {list(batch_dict.keys())}") # ['input_ids', 'attention_mask', 'position_ids', 'data_source', 'problem', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids', 'index', 'tools_kwargs', 'interaction_kwargs']
                if not self.config.meta_trainer.get("enable", False):
                    # Fallback to standard PPO if meta-training disabled
                    # self._standard_ppo_step(batch_dict)
                    continue

                # 对当前的batch循环训练outer_loop_repeat次，直到其准确率达到某个阈值或到最大迭代次数（可提前退出）
                for outer_loop_idx in range(self.config.meta_trainer.get("outer_loop_repeat", 1)):
                    if outer_loop_idx < next_outer_loop_idx:
                        continue
                    next_outer_loop_idx = -1
                    metrics = {}  # 每个 step 单独的 metrics 统计
                    timing_raw = {}
                    do_profile = (
                        self.global_steps in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(do_profile)
                    
                    with marked_timer("step", timing_raw):
                        ori_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        ori_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(ori_batch.batch))], dtype=object
                        )
                        original_size = len(ori_batch.batch) # ori_batch.batch.batch_size

                        # Step 1: 初始评估，判断哪些样本答对/答错
                        metric_name = "acc"
                        repeat_times = self.config.actor_rollout_ref.rollout.n
                        new_ori_batch, ori_prompt_uid2metric_mean = self._evaluate_current_batch(ori_batch, repeat_times=repeat_times, metric_name=metric_name) # new_ori_batch 已repeat对齐，ori_batch 未对齐

                        # # 若已达标，跳过增强
                        # if current_acc >= self.config.meta_trainer.acc_threshold:
                        #     print("[Meta-Train] Accuracy already meets threshold. Skipping augmentation.")
                        #     self._standard_ppo_step(ori_batch)
                        #     break

                        # Step 2: 场景增强生成 aug_batch，选出 n_to_augment 个prompt去增强
                        n_to_augment = int(np.ceil(2 / 3 * original_size))
                        # n_to_augment = 50
                        kept_ori_order_prompt_uids = self._get_tobe_augment_prompt_uids(ori_prompt_uid2metric_mean, n_to_augment)
                        
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(ori_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_ori_order_prompt_uids:
                                kept_traj_idxs.append(idx)
                        
                        tobe_augmented_batch = ori_batch[kept_traj_idxs]
                        target_size = self.config.meta_trainer.get("target_aug_batch_size", original_size)
                        # target_size = self.config.meta_trainer.get("target_aug_batch_size", n_to_augment+1)
                        tag_ymd, tag_ymd_hms = datetime.now().strftime('%Y%m%d'), datetime.now().strftime('%Y%m%d_%H%M%S')
                        # generator_raw_output_save_file = f"{self.config.meta_trainer.augment_data_dir}/{self.config.trainer.experiment_name}/generator_raw_output/{tag_ymd}/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}/{self.config.meta_trainer.scene_generator.model_name}_raw_output_{tag_ymd_hms}.jsonl"
                        # generator_raw_output_save_file = f"{self.config.meta_trainer.base_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_data/generator_raw_output/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}/{self.config.meta_trainer.scene_generator.model_name}_raw_output_{tag_ymd_hms}.jsonl"
                        generator_raw_output_save_file = f"{self.config.meta_trainer.base_workspace_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_data/generator_raw_output/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}/{self.config.meta_trainer.scene_generator.model_name}_raw_output_{tag_ymd_hms}.jsonl"
                        
                        with marked_timer("generate_scene_data", timing_raw):
                            aug_data_list, history_question2variants = self._generate_augdata_until_target_size(tobe_augmented_batch, target_size, history_question2variants, generator_raw_output_save_file)
                            # aug_data_list, history_question2variants = self._generate_augdata_until_target_size_sync(tobe_augmented_batch, target_size, history_question2variants, generator_raw_output_save_file)
                        print(f"[generate scene data cost time] {timing_raw['generate_scene_data']/60} mins")
                        
                        # TODO: save history_question2variants
                        # save_file = f"{self.config.meta_trainer.base_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_history/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}_augment_history_{tag_ymd_hms}.jsonl"
                        # save_file = f"{self.config.meta_trainer.base_sharedata_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_history/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}_augment_history_{tag_ymd_hms}.jsonl"
                        save_file = f"{self.config.meta_trainer.base_workspace_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_history/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}_augment_history_{tag_ymd_hms}.json"
                        # TODO: 写一个通用的保存字典到文件的函数
                        self._save_dict_to_json(history_question2variants, save_file)
                        print(f"[Augmentation End] Generated {len(aug_data_list)} augmented samples.")
                        
                        # 将生成的数据转换成 dataprob 期望的格式并保存成 jsonl 文件（方便查看）和 parquet 文件（方便RLHFDataset接口加载）
                        converted_aug_data_list = self._convert_to_experted_data_format(aug_data_list)
                        # aug_data_save_dir = f"{self.config.meta_trainer.augment_data_dir}/{self.config.trainer.experiment_name}/parsed_and_converted_data/{tag_ymd}/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}"
                        aug_data_save_dir = f"{self.config.meta_trainer.base_workspace_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_data/parsed_and_converted_data/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}"
                        # aug_data_save_dir = f"{self.config.meta_trainer.base_sharedata_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/augment_data/parsed_and_converted_data/global_training_step_{self.global_steps}_outer_loop_{outer_loop_idx}"
                        parquet_path = self._save_data_to_jsonl_and_parquet(converted_aug_data_list, aug_data_save_dir)

                        aug_dataloader = self._create_temp_dataloader_from_path(parquet_path, len(converted_aug_data_list))
                        aug_batch = DataProto.from_single_dict(next(iter(aug_dataloader)))

                        # Step 3: 构造增强后的训练 batch
                        # TODO: 可提供多种策略
                        # strategy = self.config.meta_trainer.augmentation_strategy

                        # 评估aug_batch，计算每个prompt的准确率
                        aug_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(aug_batch.batch))], dtype=object
                        )
                        metric_name = "acc"
                        new_aug_batch, aug_prompt_uid2metric_mean = self._evaluate_current_batch(aug_batch, repeat_times=repeat_times, metric_name=metric_name)

                        # 把两个batch合起来，然后用 ModifiedGaussian 根据每个prompt的错误率映射到采样概率，然后采出train_batch_size个prompt作为训练集
                        concat_batch = DataProto.concat([new_ori_batch, new_aug_batch])
                        merged_prompt_uid2metric_mean = {**ori_prompt_uid2metric_mean, **aug_prompt_uid2metric_mean}  # ori_prompt_uid2metric_mean | aug_prompt_uid2metric_mean
                        prompt_uids = list(merged_prompt_uid2metric_mean.keys())
                        acc = np.array(list(merged_prompt_uid2metric_mean.values()))
                        wrong_rate = 1 - acc

                        # 初始化修正型高斯函数（参数可按需调整）以x0为中心，k调整锐度，c是错误率为1对应的采样概率
                        map_func = ModifiedGaussian(x0=0.5, c=0.1, k=20)
                        sample_probs = wrong_rate2sample_probs(map_func, wrong_rate) # 映射后要归一化
                        train_bsz = self.config.data.train_batch_size
                        non_zero_cnt = (sample_probs != 0).sum()
                        chosen_prompt_uids = random.choice(prompt_uids, size=min(train_bsz, non_zero_cnt), replace=False, p=sample_probs)
                        chosen_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(concat_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in chosen_prompt_uids:
                                chosen_traj_idxs.append(idx)
                        train_batch = concat_batch[chosen_traj_idxs]
                        
                        if non_zero_cnt < train_bsz: # 248 vs 256
                            if train_bsz - non_zero_cnt <= train_bsz // 4:
                                # 从 chosen_prompt_uids 中采样
                                # additional_prompt_uids = [chosen_prompt_uids[random.randint(0, len(chosen_prompt_uids))] for _ in range(train_bsz - non_zero_cnt)]
                                additional_prompt_uids = random.choice(chosen_prompt_uids, size=train_bsz-non_zero_cnt, replace=False)
                            else: 
                                # 远小于 train_bsz，从 prompt_uids 中采样   这里要再优化一下，现在的做法会导致additional_prompt_ids有id重复
                                # additional_prompt_uids = [prompt_uids[random.randint(0, len(prompt_uids))] for _ in range(train_bsz - non_zero_cnt)]
                                additional_prompt_uids = random.choice(prompt_uids, size=train_bsz-non_zero_cnt, replace=False)
                            
                            additional_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(concat_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in additional_prompt_uids:
                                    additional_traj_idxs.append(idx)
                            train_batch = DataProto.concat([train_batch, concat_batch[additional_traj_idxs]])

                        # 这里会有问题：可能有重复的uid，超过了group_size，导致后面group-based adv估计出错。
                        # 所以需要重新分配uid，每 rollout.n 条轨迹分配一个新的uid
                        assert len(train_batch.batch) % self.config.actor_rollout_ref.rollout.n == 0
                        sample_cnt = len(train_batch.batch) // self.config.actor_rollout_ref.rollout.n
                        new_uids = np.array([str(uuid.uuid4()) for _ in range(sample_cnt)], dtype=object)
                        train_batch.non_tensor_batch["uid"] = np.repeat(new_uids, repeat_times, axis=0)

                        # dump actual training data
                        # train_data_output_path = f"{self.config.meta_trainer.training_data_dir}/{self.config.trainer.experiment_name}/{tag_ymd}/global_step_{self.global_steps}_outer_loop_{outer_loop_idx}_{tag_ymd_hms}.jsonl"
                        # train_data_output_path = f"{self.config.meta_trainer.base_workspace_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/actual_training_data/global_step_{self.global_steps}_outer_loop_{outer_loop_idx}_{tag_ymd_hms}.jsonl"
                        # train_data_output_path = f"{self.config.meta_trainer.base_sharedata_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/actual_training_data/global_step_{self.global_steps}_outer_loop_{outer_loop_idx}_{tag_ymd_hms}.jsonl"
                        train_data_output_path = f"{self.config.meta_trainer.base_workspace_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/actual_training_data/global_step_{self.global_steps}_outer_loop_{outer_loop_idx}_{tag_ymd_hms}.jsonl"
                        self._save_dataproto_to_jsonl(train_batch, train_data_output_path, one_id_saved_once=True)


                        # Step 4: 训练，是否要循环进行？
                        metrics = self._standard_ppo_step(train_batch, metrics, timing_raw, True)
                        print(f"metrics after ppo training: {metrics}")

                        overall_mean_acc = acc.mean()  # 提前计算，早停判断，修正总训练步数
                        exit_threshold = self.config.meta_trainer.acc_threshold
                        is_early_exit = self._is_early_exit(overall_mean_acc, exit_threshold)
                        if is_early_exit:
                            print(f"Early stopping at step {self.global_steps} for batch {batch_idx}")
                            namespace = "loop_repeat"
                            loop_metrics = {
                                f"{namespace}/early_stop_at_step": outer_loop_idx,
                            }
                            logger.log(data=loop_metrics, step=self.global_steps)
                            # 如果outer_loop_repeat=3, 在outer_loop_index=0时就退出，这时候总的训练步数应该减少2
                            self.total_training_steps -= (self.config.meta_trainer.get("outer_loop_repeat", 1) - outer_loop_idx - 1)

                        is_last_step = self.global_steps >= self.total_training_steps

                        # validate
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        ):
                            with marked_timer("testing", timing_raw, color="green"):
                                val_metrics: dict = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                            with marked_timer("copy_to_sharedata", timing_raw):
                                tmpworkspace_path = f"{self.config.meta_trainer.base_workspace_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}"
                                sharedata_path = f"{self.config.meta_trainer.base_sharedata_data_dir}/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}"
                                fast_safe_copy_to_sharedata(tmpworkspace_path, sharedata_path)

                    with marked_timer("stop_profile", timing_raw):
                        self._stop_profiling(do_profile)

                    steps_duration = timing_raw["step"]  # 每一个训练step的总耗时
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=train_batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=train_batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=train_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=train_batch)

                    # 记录 oribatch、aug_batch、train_batch 的相关指标
                    ari_mean_acc = np.mean(list(ori_prompt_uid2metric_mean.values()))
                    aug_mean_acc = np.mean(list(aug_prompt_uid2metric_mean.values()))

                    # 添加以下统计量
                    # 1. 当前 train_batch 中每个问题 k 次尝试回答完全正确 或 完全错误的统计
                    train_prompt_uid2metric_vals = defaultdict(list)
                    for uid, metric_val in zip(
                        train_batch.non_tensor_batch["uid"], train_batch.non_tensor_batch["acc"], strict=True
                    ):
                        train_prompt_uid2metric_vals[uid].append(metric_val)
                    prompt_uid2_metric_mean = {}
                    for prompt_uid, metric_vals in train_prompt_uid2metric_vals.items():
                        prompt_uid2_metric_mean[prompt_uid] = np.mean(metric_vals)
                    all_right = np.mean([metric_mean==1.0 for metric_mean in prompt_uid2_metric_mean.values()])
                    all_wrong = np.mean([metric_mean==0.0 for metric_mean in prompt_uid2_metric_mean.values()])
                    # 2. 当前 train_batch 中来自 ori_batch 和 aug_batch 的样本分别有多少
                    data_source_lst = train_batch.non_tensor_batch["data_source"]
                    from_aug_proportion = np.mean(["generated" in data_source for data_source in data_source_lst])
                    namespace = "loop_repeat"
                    loop_metrics = {
                        f"{namespace}/ori_aug_overall_acc/mean": overall_mean_acc,
                        f"{namespace}/ori_batch_acc/mean": ari_mean_acc,
                        f"{namespace}/aug_batch_acc/mean": aug_mean_acc,
                        f"{namespace}/train_batch_acc/all_correct_proportion": all_right,
                        f"{namespace}/train_batch_acc/all_wrong_proportion": all_wrong,
                        f"{namespace}/train_batch_source/from_aug_proportion": from_aug_proportion,

                        f"{namespace}/actor/entropy": metrics["actor/entropy"],
                        # f"{namespace}/epoch": epoch,
                        # f"{namespace}/global_step": self.global_steps,
                        f"{namespace}/current_batch_idx": batch_idx,
                        f"{namespace}/current_loop_iteration": outer_loop_idx,

                    }
                    metrics.update(loop_metrics)
                    # TODO: make a canonical logger that supports various backend
                    print(f"Final metrics to log: {metrics}")
                    logger.log(data=metrics, step=self.global_steps)
                    
                    progress_bar.update(1)
                    self.global_steps += 1

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                    
                    # this is experimental and may be changed/removed in the future
                    # in favor of a general-purpose data buffer pool
                    if hasattr(self.train_dataset, "on_batch_end"):
                        # The dataset may be changed after each training batch
                        self.train_dataset.on_batch_end(batch=train_batch)
                    
                    # Step 5: 早停判断，先用上一步的结果进行判断
                    if is_early_exit: # 如果整体准确率大于0.9，则跳出外层
                        # TODO，记录提前退出事件
                        # print(f"Early stopping at step {self.global_steps-1} for batch {batch_idx}")
                        # loop_metrics = {
                        #     f"{namespace}/early_stop_at_step": outer_loop_idx,
                        # }
                        # logger.log(data=loop_metrics, step=self.global_steps-1) # 注意：此时 global_steps 已 +1，日志用上一步的 step
                        break  # 退出 outer_loop_repeat，继续下一个 batch
