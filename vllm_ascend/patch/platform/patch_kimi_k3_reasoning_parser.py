# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 XTML reasoning parser backported from vLLM."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _subseq_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


def _safe_prefix_len(text: str, markers: Sequence[str]) -> int:
    positions = [position for marker in markers if (position := text.find(marker)) >= 0]
    if positions:
        return min(positions)
    overlap = 0
    for marker in markers:
        for n in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:n]):
                overlap = max(overlap, n)
                break
    return len(text) - overlap


class KimiK3ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer)
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser constructor during construction."
            )

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking")
        if thinking is None:
            thinking = chat_kwargs.get("enable_thinking", True)
        self._thinking_enabled = bool(thinking)

        self._think_open = "<|open|>think<|sep|>"
        self._think_close = "<|close|>think<|sep|>"
        self._response_open = "<|open|>response<|sep|>"
        self._response_close = "<|close|>response<|sep|>"
        self._message_close = "<|close|>message<|sep|>"
        self._end_of_msg = "<|end_of_msg|>"

        self._think_open_ids = tokenizer.encode(self._think_open, add_special_tokens=False)
        self._think_close_ids = tokenizer.encode(self._think_close, add_special_tokens=False)
        self._end_of_msg_ids = tokenizer.encode(self._end_of_msg, add_special_tokens=False)
        self._last_streaming_delta_token_ids: tuple[int, ...] | None = None
        self._last_streaming_content_token_ids: list[int] | None = None
        self._stream_buffer = ""
        self._stream_started = False
        self._stream_finished = not self._thinking_enabled

    @property
    def reasoning_start_str(self) -> str | None:
        return self._think_open

    @property
    def reasoning_end_str(self) -> str | None:
        return self._think_close

    def adjust_request(
        self,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._thinking_enabled:
            return True
        last_close = _subseq_index(input_ids, self._think_close_ids)
        last_open = _subseq_index(input_ids, self._think_open_ids)
        last_end = _subseq_index(input_ids, self._end_of_msg_ids)
        if last_end > last_open:
            return True
        if last_open == -1:
            return last_close != -1
        return last_close > last_open

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        if self._stream_finished:
            return True
        delta_ids = list(delta_ids)
        overlap = max(len(self._think_close_ids), len(self._end_of_msg_ids)) - 1
        window = input_ids[-(len(delta_ids) + overlap) :]
        return _subseq_index(window, self._think_close_ids) >= 0 or _subseq_index(window, self._end_of_msg_ids) >= 0

    def _extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        idx = _subseq_index(input_ids, self._think_close_ids)
        if idx == -1:
            return []
        return input_ids[idx + len(self._think_close_ids) :]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        cached_delta_ids = self._last_streaming_delta_token_ids
        cached_content_ids = self._last_streaming_content_token_ids
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if cached_delta_ids == tuple(input_ids) and cached_content_ids is not None:
            return cached_content_ids
        return self._extract_content_ids(input_ids)

    def _strip_content_wrapper(self, text: str) -> str:
        response_open = text.find(self._response_open)
        start = response_open + len(self._response_open) if response_open >= 0 else 0
        boundaries = [
            position
            for marker in (
                self._response_close,
                self._message_close,
                self._end_of_msg,
            )
            if (position := text.find(marker, start)) >= 0
        ]
        end = min(boundaries) if boundaries else len(text)
        return text[start:end]

    @staticmethod
    def _should_preserve_tool_channels(
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> bool:
        return bool(getattr(request, "tools", None)) and (getattr(request, "tool_choice", None) != "none")

    def _content_after_reasoning(
        self,
        text: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> str | None:
        if self._should_preserve_tool_channels(request):
            return text or None
        return self._strip_content_wrapper(text) or None

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not self._thinking_enabled:
            return None, self._content_after_reasoning(model_output, request)

        open_position = model_output.find(self._think_open)
        start = open_position + len(self._think_open) if open_position >= 0 else 0
        close_position = model_output.find(self._think_close, start)
        end_position = model_output.find(self._end_of_msg, start)
        if close_position >= 0 and (end_position < 0 or close_position < end_position):
            reasoning = model_output[start:close_position]
            rest = model_output[close_position + len(self._think_close) :]
            return reasoning or None, self._content_after_reasoning(rest, request)
        end = end_position if end_position >= 0 else len(model_output)
        return model_output[start:end] or None, None

    def _feed_reasoning(self, delta_text: str) -> DeltaMessage | None:
        self._stream_buffer += delta_text
        if not self._stream_started:
            if self._stream_buffer.startswith(self._think_open):
                self._stream_buffer = self._stream_buffer[len(self._think_open) :]
                self._stream_started = True
            elif self._think_open.startswith(self._stream_buffer):
                return None
            else:
                self._stream_started = True

        close_position = self._stream_buffer.find(self._think_close)
        end_position = self._stream_buffer.find(self._end_of_msg)
        if close_position >= 0 and (end_position < 0 or close_position < end_position):
            reasoning = self._stream_buffer[:close_position]
            content = self._stream_buffer[close_position + len(self._think_close) :]
            self._stream_buffer = ""
            self._stream_finished = True
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )
        if end_position >= 0:
            reasoning = self._stream_buffer[:end_position]
            self._stream_buffer = ""
            self._stream_finished = True
            return DeltaMessage(reasoning=reasoning) if reasoning else None

        safe_len = _safe_prefix_len(
            self._stream_buffer,
            (self._think_close, self._end_of_msg),
        )
        if not safe_len:
            return None
        reasoning = self._stream_buffer[:safe_len]
        self._stream_buffer = self._stream_buffer[safe_len:]
        return DeltaMessage(reasoning=reasoning)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text)

        del previous_text, current_text, previous_token_ids
        was_finished = self._stream_finished
        delta = self._feed_reasoning(delta_text)
        if not was_finished and self._stream_finished:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._extract_content_ids(list(current_token_ids))
        return delta

    def finish_streaming(self, current_text: str) -> DeltaMessage | None:
        del current_text
        if not self._thinking_enabled or self._stream_finished:
            return None
        tail = self._stream_buffer
        self._stream_buffer = ""
        self._stream_finished = True
        return DeltaMessage(reasoning=tail) if tail else None


ReasoningParserManager.register_module(
    name="kimi_k3",
    module=KimiK3ReasoningParser,
    force=True,
)
