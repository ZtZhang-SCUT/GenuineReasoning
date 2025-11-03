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
import asyncio
import heapq
import json
import logging
import time
from typing import Any
from uuid import uuid4
import threading

import aiohttp
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai.types.chat.chat_completion import ChatCompletion
from verl.utils.multi_scene_data.utils import _parse_json

from verl.utils.multi_scene_data.prompt_constants import MULTI_ROUND_PROMPT_TEMPLATE, MULTI_ROUND_PROMPT_TEMPLATE_EN, PROMPT_TEMPLATE, PROMPT_TEMPLATE_EN
from datetime import datetime


logger = logging.getLogger(__file__)


class SceneGeneratorCallback:
    """场景生成回调处理器，仅负责副作用：解析、保存、日志等，不返回结果"""

    def __init__(self, save_dir: str = "./", scheduler: "ChatCompletionScheduler"=None):
        self.save_dir = Path(save_dir)
        self.scheduler = scheduler
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.id_to_result = {}

    def _parse_output(self, raw_output: str) -> Optional[Dict[str, str]]:
        """解析模型输出为结构化结果"""
        try:
            text = raw_output.strip()
            # 在去除 ```json 后、解析前，加入：
            text = text.replace('\x0c', '\\f').replace('\x08', "\\b")  # 把 form feed 换回 \f
            parsed = _parse_json(text)
            if parsed and "new_question" in parsed and "solution" in parsed:
                return {
                    "question": parsed["new_question"],
                    "solution": parsed["solution"],
                    "explanation": parsed.get("explanation", ""),
                }
            else:
                logger.warning(f"Missing required fields in output: {text[:200]}...")
                return None
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f"JSON parse failed: {e}")
            return None

    # def _fix_invalid_json(self, text: str) -> str:
    #     import re
    #     def repl(match):
    #         prefix, bad_char = match.group(1), match.group(2)
    #         return prefix + ("\\" + bad_char if len(prefix) % 2 == 1 else bad_char)
    #     return re.sub(r'(\\+)(?!["\\/bfnrtu])(.?)', repl, text)

    def __call__(self, messages: list[dict[str, str]], completions: ChatCompletion, info: dict[str, Any]):
        """生成完成后调用，执行保存等操作"""
        # print(f"model completions: {completions}")
        # print(f"id: {completions.id}\nchoices: {completions.choices[0]}\nmodel: {completions.model}\nusage: {completions.usage}")
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        raw_output = message["content"].strip()
        parsed_output = self._parse_output(raw_output)
        
        # self.scheduler.set_parsed_output(completions.id, parsed_output)  # 待实现
        request_id = completions.id.removeprefix("chatcmpl-")
        # print(f"current request_id: {request_id}")
        self.id_to_result[request_id] = parsed_output
        
        # 保存
        save_file = info.get("save_file", "")
        is_save = info.get("is_save", False)
        if is_save:
            if not save_file:
                ts = info.get("timestamp", str(uuid.uuid4()))
                save_file = self.save_dir / f"scene_{ts}.jsonl"

            save_path = Path(save_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            info.update({
                "time": datetime.now().strftime('%Y%m%d_%H%M%S')
            })
            record = {
                "messages": messages,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "info": {k: v for k, v in info.items() if not "save" in k and not k.startswith("__")},
            }

            with save_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if parsed_output:
                logger.info(f"✅ Saved result to {save_file}")
            else:
                logger.warning(f"⚠️ Saved failed result to {save_file}")

        else:
            logger.info(f"Save file is not provided, no need to saved result")

    def postprocess(self, ids: list[str]) -> list[dict[str, str]]:
        # 可以进行质量检查等步骤
        # print(self.id_to_result.keys())
        return [self.id_to_result.pop(id) for id in ids]


class ChatCompletionScheduler:
    def __init__(
        self,
        model_name: str,
        api_urls: list[str],
        api_keys: list[str],
        api_timeout: int = 180,
        max_retries: int = 3,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.model_name = model_name

        # Least requests load balancing
        self.api_configs = [[0, url, key] for url, key in zip(api_urls, api_keys)]
        heapq.heapify(self.api_configs)

        # LRU cache to map request_id to address
        # self.request_id_to_address = LRUCache(maxsize=max_cache_size)
        self.timeout = api_timeout

        # 重试装饰器配置
        self._retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
            before_sleep=before_sleep_log(logging.getLogger(), logging.INFO)
        )

        self.background_tasks = set()

        self.completion_callback = SceneGeneratorCallback(scheduler=self)

    def submit_chat_completions(self, *, messages: list[dict[str, str]], request_id: str, info: dict[str, Any]):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info))

        # “fire-and-forget” background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_chat_completions_and_callback(
        self,
        messages: list[dict[str, str]],
        request_id: str,
        info: dict[str, Any],
    ):
        """Submit chat completion request, wait request finish and do callback."""
        
        # 选择服务器
        config = self.api_configs[0]
        url_and_key = {
            "api_url": config[1], 
            "api_key": config[2]
        }
        config[0] += 1
        heapq.heapreplace(self.api_configs, config)

        completions, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp_with_retry(
                url_and_key,
                messages=messages,
                extra_headers={"Content-Type": "application/json", "x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    @property
    def call_api_with_retry(self):
        # 返回一个装饰了重试逻辑的函数
        return self._retry_decorator(self._chat_completions_aiohttp)
    
    async def _chat_completions_aiohttp_with_retry(self, url_and_key: dict, **chat_complete_request) -> str:
        return await self.call_api_with_retry(url_and_key, **chat_complete_request)

    async def _chat_completions_aiohttp(self, url_and_key: dict, **chat_complete_request) -> ChatCompletion:
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            session = aiohttp.ClientSession(timeout=timeout)
            # print(f"chat_complete_request: {chat_complete_request}")
            async with session.post(
                # url=f"http://{address}/v1/chat/completions",
                url=f"{url_and_key['api_url']}",
                headers={"Authorization": f"{url_and_key['api_key']}", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, batch: list[list[dict[str, str]]], **kwargs) -> list[dict]:
        t_start = time.time()
        sampling_params = dict(
            model=self.model_name,
        )
        sampling_keys = {"temperature", "top_p", "max_tokens", "presence_penalty"}
        sampling_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sampling_keys
        }

        sampling_params.update(sampling_kwargs)
        
        other_info = {      # {"save_file": xxx}
            k: v for k, v in kwargs.items()
            if k not in sampling_keys
        }

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {sampling_params}")
        
        tasks, batch_conversations = [], [None] * len(batch)
        # use new request_id to avoid duplicate request_id problem
        request_ids = [uuid4().hex for _ in range(len(batch))]
        for batch_index, batch_item in enumerate(batch):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            # batch_conversations[batch_index] = batch_item["messages"]
            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_item,
                        request_id=request_ids[batch_index],
                        sampling_params=sampling_params,
                        other_info=other_info
                    )
                )
            )

        await asyncio.gather(*tasks)
        output_batch = self.completion_callback.postprocess(request_ids)
        print("[ChatCompletionScheduler] generate_sequences done")
        return output_batch

    async def _submit_chat_completions_semaphore(
        self, messages: list[dict[str, str]], request_id: str, sampling_params: dict[str, Any], other_info: dict[str, str]
    ):
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
        }

        info.update(other_info)

        self.submit_chat_completions(messages=messages, request_id=request_id, info=info)

        # Wait until all completion requests are done
        await done.wait()


class SceneGeneratorManager:
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    def __init__(
            self, 
            model_name: str, 
            api_urls: List[str],
            api_keys: List[str],
            max_retries: int = 3,
            timeout: float = 180,
        ):
        """Initialize AsyncLLMServerManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
        """
        self.model_name = model_name

        self.api_urls = api_urls
        self.api_keys = api_keys
        self.max_retries = max_retries
        self.api_timeout = timeout

        # Init user provided chat scheduler in sperate thread.
        self.chat_scheduler: ChatCompletionScheduler = None
        self.chat_scheduler_exception: Exception = None
        self.chat_scheduler_loop = None
        self.chat_scheduler_ready = threading.Event()
        self.chat_scheduler_thread = threading.Thread(target=self._init_chat_scheduler, daemon=True)
        self.chat_scheduler_thread.start()
        self.chat_scheduler_ready.wait()

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)

        try:
            self.chat_scheduler = ChatCompletionScheduler(
                model_name=self.model_name,
                api_urls=self.api_urls,
                api_keys=self.api_keys,
                api_timeout=self.api_timeout,
                max_retries=self.max_retries
            )
        except Exception as e:
            logger.exception(f"chat_scheduler init error: {e}")
            self.chat_scheduler_exception = e
        finally:
            self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()

    def submit_chat_completions(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any],
    ):
        """Submit a chat completion request to chat scheduler and wait until it is done.
        To submit multiple requests in parallel, please use `generate_sequences` instead.

        Args: same as ChatCompletionScheduler.submit_chat_completions.
        """
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler._submit_chat_completions_semaphore(
                messages=messages,
                request_id=None,
                sampling_params=sampling_params,
            ),
            self.chat_scheduler_loop,
        )
        future.result()

    def _format_previous_aug(self, previous_aug_questions: List[str]) -> str:
        """格式化之前生成的问题"""
        if not previous_aug_questions: return ""
        formatted = []
        for i, aug_question in enumerate(previous_aug_questions, 1):
            formatted.append(f"{i}. {aug_question}")
        return "\n".join(formatted)
    
    def _build_messages(self, original_question: str, previous: List[str]) -> List[Dict[str, str]]:
        if previous:
            aug = self._format_previous_aug(previous)
            prompt = MULTI_ROUND_PROMPT_TEMPLATE_EN.format(
                previous_aug=aug, original_question=original_question
            )
        else:
            prompt = PROMPT_TEMPLATE_EN.format(
                original_question=original_question
            )
        return [{"role": "user", "content": prompt}]
    
    def generate_batch(self, prompt_context_lst: list[dict], **kwargs) -> list[dict]:
        """Generate multiple sequences in parallel via chat scheduler."""
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        messages_lst = []
        for prompt_context in prompt_context_lst:
            messages_lst.append(
                self._build_messages(prompt_context["original_question"], prompt_context["previous_aug_questions"])
            )
        # print(f"messages_lst: {messages_lst}")

        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler.generate_sequences(messages_lst, **kwargs), self.chat_scheduler_loop
        )
        return future.result()

if __name__ == "__main__":
    manager = SceneGeneratorManager(
        model_name="qwen3-235b-a22b",
        api_urls=["http://172.31.215.100:8001/v1/chat/completions"],
        api_keys=["fake-key"]
    )

    # prompt_contexts = [
    #     {"original_question": "How to calculate distance?", "previous_aug_questions": []}
    # ]

    # results = manager.generate_sequences(prompt_contexts, temperature=0.7, top_p=0.9)
    # print(f"final result: \n{results}")

    # prompt_contexts = [
    #     {"original_question": "What is 2+2?", "previous_aug_questions": []}
    # ]
    # save_file = "./my_outputs/temp.jsonl"

    # results = manager.generate_sequences(prompt_contexts, save_file=str(save_file), is_save=True)
    # print(f"final result: \n{results}")

    prompt_contexts = [
        {"original_question": "How to calculate distance?", "previous_aug_questions": []},
        {"original_question": "What is 2+2?", "previous_aug_questions": []},
    ]
    results = manager.generate_batch(prompt_contexts, save_file="./test.jsonl", is_save=True)
    print(f"final result: \n{results}")

