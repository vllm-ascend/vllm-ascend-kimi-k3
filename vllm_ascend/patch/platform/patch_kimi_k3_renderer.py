#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Render Kimi K3 chat prompts with its tokenizer-owned Python encoder.

Kimi K3 deliberately does not publish a Jinja chat template. Its trusted
remote ``TikTokenTokenizer.apply_chat_template`` implements the XTML protocol,
including typed tool calls, reasoning controls, and multimodal placeholders.
The regular HF renderer rejects tokenizers without a Jinja template, so K3
uses a dedicated ``kimi_k3`` mode backed by the standard HF tokenizer loader.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.renderers import registry as renderer_registry
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.params import ChatParams
from vllm.tokenizers.hf import HfTokenizer
from vllm.tokenizers.registry import TokenizerRegistry
from vllm.utils.async_utils import make_async

KIMI_K3_MODEL_TYPE = "kimi_k3"
KIMI_K3_RENDERER_MODE = "kimi_k3"
KIMI_K3_IMAGE_PROMPT = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
KIMI_K3_PROMPT_TOOL_CHOICE_KEY = "_kimi_k3_prompt_tool_choice"
_KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX = "kimi_k3:"
_KIMI_K3_PROMPT_TOOL_CHOICES = frozenset({"none", "auto", "required"})
_K3_THINKING_EFFORTS = frozenset({"low", "high", "max"})
_ORIGINAL_TOKENIZER_ARGS_ATTR = "_ascend_original_kimi_k3_tokenizer_args_from_config"
_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {"image": {"image_mode": None}}


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def encode_kimi_k3_prompt_tool_choice(tool_choice: str) -> str:
    if tool_choice not in _KIMI_K3_PROMPT_TOOL_CHOICES:
        raise ValueError(f"Unsupported Kimi K3 prompt tool choice: {tool_choice!r}.")
    return _KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX + tool_choice


def decode_kimi_k3_prompt_tool_choice(encoded_choice: str) -> str:
    if not encoded_choice.startswith(_KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX):
        raise ValueError("Malformed Kimi K3 prompt tool choice.")
    tool_choice = encoded_choice[len(_KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX) :]
    if tool_choice not in _KIMI_K3_PROMPT_TOOL_CHOICES:
        raise ValueError(f"Unsupported Kimi K3 prompt tool choice: {tool_choice!r}.")
    return tool_choice


def is_kimi_k3_model_config(model_config: ModelConfig) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == KIMI_K3_MODEL_TYPE


def _normalize_developer_messages(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Convert developer roles without reordering their content or tools."""

    converted: list[ConversationMessage] = []
    for message in conversation:
        if message["role"] == "developer":
            converted_message = dict(message)
            converted_message["role"] = "system"
            converted.append(converted_message)  # type: ignore[arg-type]
        else:
            converted.append(message)
    return converted


def _trusted_image_prompts(
    conversation: list[ConversationMessage],
) -> list[str]:
    """Build server-owned image prompts for structured OpenAI content parts."""

    image_prompts: list[str] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "image",
                "image_url",
            ):
                image_prompts.append(KIMI_K3_IMAGE_PROMPT)
    return image_prompts


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Renderer that delegates the complete chat protocol to K3's tokenizer."""

    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template,
            executor=self._executor,
        )

    def _apply_chat_template(
        self,
        conversation_data: list[ConversationMessage],
        **kwargs: Any,
    ) -> list[int]:
        # K3's Python encoder is the source of truth. In particular, do not let
        # an optional server/request Jinja value replace or filter its kwargs.
        prompt_tool_choice = kwargs.pop(KIMI_K3_PROMPT_TOOL_CHOICE_KEY, None)
        if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
            kwargs.setdefault("thinking", enable_thinking)
        if (reasoning_effort := kwargs.pop("reasoning_effort", None)) is not None:
            kwargs.setdefault("thinking_effort", reasoning_effort)
        thinking = kwargs.get("thinking", True)
        if not isinstance(thinking, bool):
            raise VLLMValidationError(
                "Kimi K3 thinking must be a boolean.",
                parameter="chat_template_kwargs.thinking",
                value=thinking,
            )
        thinking_effort = kwargs.get("thinking_effort")
        if thinking and thinking_effort is not None and thinking_effort not in _K3_THINKING_EFFORTS:
            raise VLLMValidationError(
                "Kimi K3 reasoning_effort supports low, high, and max.",
                parameter="reasoning_effort",
                value=thinking_effort,
            )
        for protected_key in (
            "add_generation_prompt",
            "chat_template",
            "continue_final_message",
            "conversation",
            "image_prompts",
            "max_length",
            "padding",
            "return_dict",
            "return_tensors",
            "tokenize",
            "truncation",
        ):
            kwargs.pop(protected_key, None)
        if prompt_tool_choice is not None:
            # vLLM 0.23 treats the literal string ``auto`` as an unset value
            # while merging ChatParams defaults. The private typed key survives
            # that merge so a server default cannot turn an auto request into
            # a required/none prompt.
            kwargs["tool_choice"] = decode_kimi_k3_prompt_tool_choice(prompt_tool_choice)
        if kwargs.get("tool_choice") == "none" and not kwargs.get("tools"):
            kwargs.pop("tool_choice")
        prompt = self.get_tokenizer().apply_chat_template(
            conversation=conversation_data,
            tokenize=True,
            add_generation_prompt=True,
            image_prompts=_trusted_image_prompts(conversation_data),
            padding=False,
            truncation=False,
            return_tensors=None,
            return_dict=False,
            **kwargs,
        )
        if not isinstance(prompt, list) or any(not isinstance(token_id, int) for token_id in prompt):
            raise TypeError("Kimi K3 tokenizer must return a flat list of token IDs.")
        return prompt

    def _render_conversation(
        self,
        conversation: list[ConversationMessage],
        mm_data,
        mm_uuids,
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation = _normalize_developer_messages(conversation)
        prompt_raw = self._apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return conversation, prompt

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        return self._render_conversation(
            conversation,
            mm_data,
            mm_uuids,
            params,
        )

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        conversation = _normalize_developer_messages(conversation)

        prompt_raw = await self._apply_chat_template_async(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt


if not hasattr(renderer_registry, _ORIGINAL_TOKENIZER_ARGS_ATTR):
    setattr(
        renderer_registry,
        _ORIGINAL_TOKENIZER_ARGS_ATTR,
        renderer_registry.tokenizer_args_from_config,
    )


@wraps(getattr(renderer_registry, _ORIGINAL_TOKENIZER_ARGS_ATTR))
def _tokenizer_args_with_kimi_k3_renderer(model_config: ModelConfig, **kwargs):
    original = getattr(renderer_registry, _ORIGINAL_TOKENIZER_ARGS_ATTR)
    tokenizer_args = original(model_config, **kwargs)
    renderer_mode, *remaining_args = tokenizer_args

    # Preserve the standard HF tokenizer loader and explicit non-HF modes. K3
    # only needs a different renderer when auto/slow/hf resolves to HF.
    if renderer_mode == "hf" and is_kimi_k3_model_config(model_config):
        renderer_mode = KIMI_K3_RENDERER_MODE

    return renderer_mode, *remaining_args


renderer_registry.tokenizer_args_from_config = _tokenizer_args_with_kimi_k3_renderer

if KIMI_K3_RENDERER_MODE not in TokenizerRegistry.tokenizers:
    TokenizerRegistry.register(
        KIMI_K3_RENDERER_MODE,
        "vllm.tokenizers.hf",
        "CachedHfTokenizer",
    )

if KIMI_K3_RENDERER_MODE not in renderer_registry.RENDERER_REGISTRY.renderers:
    renderer_registry.RENDERER_REGISTRY.register(
        KIMI_K3_RENDERER_MODE,
        __name__,
        "KimiK3Renderer",
    )
