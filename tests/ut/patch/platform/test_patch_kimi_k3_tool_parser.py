# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.exceptions import VLLMValidationError
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import DelegatingParser
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm_ascend.patch.platform.patch_kimi_k3_reasoning_parser import (
    KimiK3ReasoningParser,
)
from vllm_ascend.patch.platform.patch_kimi_k3_tool_parser import (
    KimiK3ToolParser,
    get_kimi_k3_structural_tag,
)

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_CLOSE = f"{CLOSE}response{SEP}"
MESSAGE_CLOSE = f"{CLOSE}message{SEP}"
END_OF_MSG = "<|end_of_msg|>"


class DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == THINK_OPEN:
            return [1, 2, 3]
        if text == THINK_CLOSE:
            return [4, 2, 3]
        return [ord(ch) for ch in text]


class KimiK3DelegatingParser(DelegatingParser):
    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = KimiK3ToolParser


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )


def _named_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "calc"}},
    )


def _arg(key: str, typ: str, value: str) -> str:
    return f'{OPEN}argument key="{key}" type="{typ}"{SEP}{value}{CLOSE}argument{SEP}'


def _call(tool: str, index: int, *args: str) -> str:
    body = "".join(args)
    return f'{OPEN}call tool="{tool}" index="{index}"{SEP}{body}{CLOSE}call{SEP}'


def _response(content: str) -> str:
    return f"{OPEN}response{SEP}{content}{RESPONSE_CLOSE}"


def _tools(*calls: str) -> str:
    return f"{OPEN}tools{SEP}{''.join(calls)}{CLOSE}tools{SEP}"


def _chat_serving() -> OpenAIServingChat:
    serving = object.__new__(OpenAIServingChat)
    serving.parser_cls = KimiK3DelegatingParser
    serving.tool_parser = KimiK3ToolParser
    serving.tool_call_id_type = "random"
    serving.response_role = "assistant"
    serving.use_harmony = False
    serving.enable_auto_tools = True
    serving.enable_force_include_usage = False
    serving.enable_prompt_tokens_details = False
    serving.system_fingerprint = None
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.request_logger = None
    return serving


def _serve(
    request: ChatCompletionRequest,
    text: str,
    *,
    finish_reason: str = "stop",
    kv_transfer_params: dict | None = None,
):
    result = RequestOutput(
        request_id="test-request",
        prompt="prompt",
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=list(range(1, len(text) + 1)),
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=True,
        kv_transfer_params=kv_transfer_params,
    )
    serving = _chat_serving()
    metadata = RequestResponseMetadata(request_id="test-request")

    async def results():
        yield result

    async def run():
        if request.stream:
            return [
                json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
                async for chunk in serving.chat_completion_stream_generator(
                    request,
                    results(),
                    "test-request",
                    request.model,
                    [],
                    DummyTokenizer(),
                    metadata,
                    chat_template_kwargs={"thinking": False},
                )
                if chunk != "data: [DONE]\n\n"
            ]

        parser = KimiK3DelegatingParser(
            DummyTokenizer(),
            request.tools,
            chat_template_kwargs={"thinking": False},
        )
        return await serving.chat_completion_full_generator(
            request,
            results(),
            "test-request",
            request.model,
            [],
            DummyTokenizer(),
            metadata,
            parser,
        )

    return asyncio.run(run())


def test_extract_tool_calls_with_response_and_typed_arguments():
    parser = KimiK3ToolParser(DummyTokenizer())

    output = _response("answer") + _tools(
        _call(
            "calc",
            1,
            _arg("x", "number", "1"),
            _arg("flag", "boolean", "true"),
            _arg("text", "string", "raw"),
        )
    )
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.content == "answer"
    assert len(extracted.tool_calls) == 1
    tool_call = extracted.tool_calls[0]
    assert tool_call.id == "calc:0"
    assert tool_call.function.name == "calc"
    assert json.loads(tool_call.function.arguments) == {
        "x": 1,
        "flag": True,
        "text": "raw",
    }


def test_typed_arguments_preserve_order_and_number_formatting():
    parser = KimiK3ToolParser(DummyTokenizer())
    output = _tools(
        _call(
            "calc",
            1,
            _arg("y", "number", "1.0"),
            _arg("x", "number", "2"),
            _arg("items", "array", '["left","right"]'),
        )
    )

    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tool_calls[0].function.arguments == ('{"y":1.0,"x":2,"items":["left","right"]}')


def test_extract_tool_calls_preserves_raw_json_block():
    parser = KimiK3ToolParser(DummyTokenizer())
    raw = '{"b": 1,  "a": [2 , 3]}'
    json_block = f'{OPEN}json type="object"{SEP}{raw}{CLOSE}json{SEP}'

    extracted = parser.extract_tool_calls(
        _response("") + _tools(_call("calc", 1, json_block)),
        _request(),
    )

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.arguments == raw


def test_extract_tool_calls_ignores_epilogue_and_post_message_text():
    parser = KimiK3ToolParser(DummyTokenizer())
    output = "answer" + RESPONSE_CLOSE + "\n" + _tools(_call("calc", 1)) + "\n" + MESSAGE_CLOSE + "junk" + END_OF_MSG

    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.content == "answer"
    assert len(extracted.tool_calls) == 1


def test_extract_tool_calls_keeps_complete_calls_from_truncated_tools_channel():
    parser = KimiK3ToolParser(DummyTokenizer())

    extracted = parser.extract_tool_calls(
        f"{OPEN}tools{SEP}" + _call("calc", 1),
        _request(),
    )

    assert [call.id for call in extracted.tool_calls] == ["calc:0"]


def test_delegating_parser_preserves_tool_calls_after_reasoning():
    parser = KimiK3DelegatingParser(DummyTokenizer())
    output = f"{THINK_OPEN}step{THINK_CLOSE}" + _response("answer") + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output,
        _request(),
        enable_auto_tools=True,
    )

    assert reasoning == "step"
    assert content == "answer"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "calc:0"
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_required_tool_choice_uses_xtml_parser():
    parser = KimiK3DelegatingParser(DummyTokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})
    output = f"{THINK_OPEN}step{THINK_CLOSE}" + _response("") + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output,
        request,
        enable_auto_tools=False,
    )

    assert reasoning == "step"
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_named_tool_choice_uses_xtml_parser():
    parser = KimiK3DelegatingParser(DummyTokenizer())
    output = f"{THINK_OPEN}step{THINK_CLOSE}" + _response("") + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output,
        _named_request(),
        enable_auto_tools=False,
    )

    assert reasoning == "step"
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_auto_no_call_strips_consumed_response_prefix():
    parser = KimiK3DelegatingParser(DummyTokenizer(), chat_template_kwargs={"thinking": False})
    request = _request().model_copy(update={"chat_template_kwargs": {"thinking": False}})

    reasoning, content, tool_calls = parser.parse(
        f"answer{RESPONSE_CLOSE}",
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "answer"
    assert tool_calls is None


def test_delegating_parser_required_call_strips_consumed_response_prefix():
    parser = KimiK3DelegatingParser(DummyTokenizer(), chat_template_kwargs={"thinking": False})
    request = _request().model_copy(
        update={
            "tool_choice": "required",
            "chat_template_kwargs": {"thinking": False},
        }
    )
    output = RESPONSE_CLOSE + _tools(_call("calc", 1, _arg("x", "number", "1")))

    reasoning, content, tool_calls = parser.parse(
        output,
        request,
        enable_auto_tools=False,
    )

    assert reasoning is None
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "calc"
    assert json.loads(tool_calls[0].arguments) == {"x": 1}


def test_delegating_parser_rejects_truncated_tool_call():
    parser = KimiK3DelegatingParser(DummyTokenizer(), chat_template_kwargs={"thinking": False})
    request = _request().model_copy(
        update={
            "tool_choice": "required",
            "chat_template_kwargs": {"thinking": False},
        }
    )

    with pytest.raises(ValueError, match="incomplete Kimi K3 tool call"):
        parser.parse(
            (f'{RESPONSE_CLOSE}{OPEN}tools{SEP}{OPEN}call tool="calc" index="1"'),
            request,
            enable_auto_tools=False,
        )


def test_extract_tool_calls_unescapes_attributes():
    parser = KimiK3ToolParser(DummyTokenizer())

    output = _tools(_call("a&amp;b&quot;c", 1, _arg("k&amp;q", "string", "v")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == 'a&b"c'
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"k&q": "v"}


def test_extract_tool_calls_allows_less_than_in_attributes():
    parser = KimiK3ToolParser(DummyTokenizer())

    output = _tools(_call("calc<beta", 1, _arg("foo<bar", "string", "raw")))
    extracted = parser.extract_tool_calls(output, _request())

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == "calc<beta"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"foo<bar": "raw"}


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_streaming_split_markers_do_not_leak(chunk_size):
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request()
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    output = _response("Hi") + _tools(
        _call("calc", 1, _arg("x", "number", "1")),
        _call("calc", 2, _arg("x", "number", "2")),
    )
    chunks = [output[start : start + chunk_size] for start in range(0, len(output), chunk_size)]

    for i, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [i]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[i],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    content = "".join(message.content or "" for message in messages)
    tool_deltas = [tool_call for message in messages for tool_call in (message.tool_calls or [])]

    assert content == "Hi"
    assert OPEN not in content
    assert SEP not in content
    assert len(tool_deltas) == 2
    assert tool_deltas[0].id == "calc:0"
    assert tool_deltas[0].function.name == "calc"
    assert json.loads(tool_deltas[0].function.arguments) == {"x": 1}
    assert tool_deltas[1].id == "calc:1"
    assert json.loads(tool_deltas[1].function.arguments) == {"x": 2}


def test_streaming_consumed_response_prefix_no_call_keeps_content():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request()
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    chunks = ["O", "K", CLOSE, f"response{SEP}"]

    for i, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [i]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[i],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    assert "".join(message.content or "" for message in messages) == "OK"
    assert all(CLOSE not in (message.content or "") for message in messages)


def test_streaming_finish_flushes_partial_marker_as_content():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(),
        chat_template_kwargs={"thinking": False},
    )
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tool_choice="none",
        chat_template_kwargs={"thinking": False},
    )
    messages = []

    for token_id, (chunk, finished) in enumerate(
        (("answer", False), ("<|clo", True)),
        start=1,
    ):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[token_id],
            request=request,
            prompt_token_ids=[1],
            finished=finished,
        )
        if delta is not None:
            messages.append(delta)

    assert "".join(message.content or "" for message in messages) == "answer<|clo"


def test_delegating_parser_tool_choice_none_strips_response_markers():
    parser = KimiK3DelegatingParser(DummyTokenizer(), chat_template_kwargs={"thinking": False})
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tool_choice="none",
        chat_template_kwargs={"thinking": False},
    )

    messages = []
    for token_id, chunk in enumerate(("OK", CLOSE, f"response{SEP}"), start=1):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[token_id],
            request=request,
            prompt_token_ids=[1],
            finished=False,
        )
        if delta is not None:
            messages.append(delta)

    assert "".join(message.content or "" for message in messages) == "OK"
    assert all(not message.tool_calls for message in messages)


def test_delegating_parser_auto_streaming_uses_v023_signature():
    parser = KimiK3DelegatingParser(
        DummyTokenizer(),
        chat_template_kwargs={"thinking": False},
    )
    request = _request().model_copy(update={"chat_template_kwargs": {"thinking": False}})

    delta = parser.parse_delta(
        delta_text="answer",
        delta_token_ids=[1],
        request=request,
        prompt_token_ids=[1],
        finished=False,
    )

    assert delta is not None
    assert delta.content == "answer"


@pytest.mark.parametrize(
    "tool_choice",
    ["required", _named_request().tool_choice],
)
def test_serving_full_preserves_forced_content(tool_choice):
    request = _request().model_copy(update={"tool_choice": tool_choice})
    output = _response("Calling the selected tool.") + _tools(_call("calc", 1))

    response = _serve(request, output)

    assert not isinstance(response, ErrorResponse)
    assert response.choices[0].message.content == "Calling the selected tool."
    assert [call.function.name for call in response.choices[0].message.tool_calls] == ["calc"]


def test_serving_full_preserves_length_finish_reason():
    response = _serve(
        _request(),
        _response("") + _tools(_call("calc", 1)),
        finish_reason="length",
    )

    assert not isinstance(response, ErrorResponse)
    assert response.choices[0].finish_reason == "length"


def test_serving_full_skips_parser_for_remote_decode(monkeypatch):
    request = _request().model_copy(
        update={
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
            }
        }
    )

    def fail_parse(*args, **kwargs):
        pytest.fail("Kimi K3 parser must not run for remote decode requests")

    monkeypatch.setattr(KimiK3DelegatingParser, "parse", fail_parse)
    response = _serve(
        request,
        "internal-prefill-token",
        finish_reason="length",
        kv_transfer_params={"remote_engine_id": "prefill-engine"},
    )

    assert not isinstance(response, ErrorResponse)
    assert response.kv_transfer_params == {"remote_engine_id": "prefill-engine"}


def test_serving_full_keeps_parser_for_remote_prefill(monkeypatch):
    request = _request().model_copy(
        update={
            "kv_transfer_params": {
                "do_remote_decode": False,
                "do_remote_prefill": True,
            }
        }
    )
    original_parse = KimiK3DelegatingParser.parse
    parse_calls = 0

    def count_parse(self, *args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(self, *args, **kwargs)

    monkeypatch.setattr(KimiK3DelegatingParser, "parse", count_parse)
    response = _serve(request, _response("answer"))

    assert not isinstance(response, ErrorResponse)
    assert parse_calls == 1
    assert response.choices[0].message.content == "answer"


@pytest.mark.parametrize(
    ("engine_finish_reason", "expected_finish_reason"),
    [("stop", "tool_calls"), ("length", "length")],
)
def test_serving_stream_splits_tool_delta_from_finish_reason(
    engine_finish_reason,
    expected_finish_reason,
):
    request = _request().model_copy(update={"stream": True})
    payloads = _serve(
        request,
        _response("") + _tools(_call("calc", 1)),
        finish_reason=engine_finish_reason,
    )
    choices = [choice for payload in payloads for choice in payload["choices"]]

    tool_choices = [choice for choice in choices if choice["delta"].get("tool_calls")]
    assert tool_choices
    assert all(choice["finish_reason"] is None for choice in tool_choices)
    terminal = [choice for choice in choices if choice.get("finish_reason") is not None]
    assert [choice["finish_reason"] for choice in terminal] == [expected_finish_reason]
    assert all(not choice["delta"].get("tool_calls") for choice in terminal)


def test_serving_stream_does_not_duplicate_terminal_token_ids():
    request = _request().model_copy(update={"stream": True, "return_token_ids": True})
    payloads = _serve(request, _response("") + _tools(_call("calc", 1)))
    choices = [choice for payload in payloads for choice in payload["choices"]]

    tool_choice = next(choice for choice in choices if choice["delta"].get("tool_calls"))
    terminal = next(choice for choice in choices if choice.get("finish_reason"))
    assert tool_choice["token_ids"]
    assert "token_ids" not in terminal


def test_adjust_request_keeps_xtml_markers_contiguous():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request()

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False
    assert KimiK3ToolParser.supports_required_and_named is False


def test_adjust_request_required_uses_xtml_parser_not_json_guidance():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})

    adjusted = parser.adjust_request(request)

    assert adjusted.structured_outputs is not None
    assert adjusted.structured_outputs.structural_tag is not None
    assert adjusted.structured_outputs.json is None
    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False


def test_adjust_request_named_tool_choice_uses_xtml_structural_tag():
    parser = KimiK3ToolParser(DummyTokenizer())
    adjusted = parser.adjust_request(_named_request())

    assert adjusted.structured_outputs is not None
    assert adjusted.structured_outputs.structural_tag is not None
    assert adjusted.skip_special_tokens is False


def test_adjust_request_rejects_named_tool_choice_without_structural_tag(monkeypatch):
    parser = KimiK3ToolParser(DummyTokenizer())
    monkeypatch.setattr(parser, "get_structural_tag", lambda request: None)
    with pytest.raises(VLLMValidationError) as exc_info:
        parser.adjust_request(_named_request())

    assert exc_info.value.parameter == "tool_choice"
    assert "requires strict tool calling" in str(exc_info.value)


def test_required_structural_tag_matches_k3_xtml():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})
    structural_tag = parser.get_structural_tag(request)

    assert structural_tag is not None
    grammar = Grammar.from_structural_tag(structural_tag)
    valid = _response("") + _tools(_call("calc", 1))
    assert _is_grammar_accept_string(grammar, valid)
    assert not _is_grammar_accept_string(grammar, _response("no call"))


def test_required_structural_tag_accepts_raw_json_arguments():
    parser = KimiK3ToolParser(DummyTokenizer())
    request = _request().model_copy(update={"tool_choice": "required"})
    grammar = Grammar.from_structural_tag(parser.get_structural_tag(request))
    raw = f'{OPEN}json type="object"{SEP}{{}}{CLOSE}json{SEP}'

    assert _is_grammar_accept_string(grammar, _response("") + _tools(_call("calc", 1, raw)))


def test_structural_tag_supports_union_types_and_reasoning_prefix():
    request = _request()
    request.tools[0].function.parameters = {
        "type": "object",
        "properties": {"count": {"type": ["integer", "null"]}},
        "required": ["count"],
    }
    structural_tag = get_kimi_k3_structural_tag(request.tools, "required", True)
    grammar = Grammar.from_structural_tag(structural_tag)
    output = f"step{THINK_CLOSE}{OPEN}response{SEP}{RESPONSE_CLOSE}" + _tools(
        _call("calc", 1, _arg("count", "null", "null"))
    )

    assert _is_grammar_accept_string(grammar, output)


def test_auto_without_strict_tool_is_not_constrained():
    parser = KimiK3ToolParser(DummyTokenizer())

    assert parser.get_structural_tag(_request()) is None
