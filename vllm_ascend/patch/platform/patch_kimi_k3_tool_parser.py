# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parser for the Kimi K3 (XTML) chat format.

This turns the generated XTML ``response`` and ``tools`` channels back into
OpenAI-compatible ``content`` and ``tool_calls``.

K3 assistant tool calls live in a nested ``tools`` channel::

    <|open|>tools<|sep|>
      <|open|>call tool="python" index="1"<|sep|>
        <|open|>argument key="code" type="string"<|sep|>print(1)<|close|>argument<|sep|>
        <|open|>argument key="opts" type="object"<|sep|>{"a":1}<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

where ``<|open|>``, ``<|close|>``, ``<|sep|>`` are dedicated special tokens. The plain
reply lives in a sibling ``response`` channel which is unwrapped into content.
``response`` always precedes ``tools`` (see the protocol channel grammar).

Argument decoding mirrors the template's type tagging (inverse encoding):
  * type="string"  -> value is the RAW text, no unescaping (template emits it raw)
  * other types    -> value is JSON-decoded (number/boolean/null/object/array)

Attribute values (``tool=``, ``index=``, ``key=``, ``type=``) ARE escaped on the
encode side (``&`` -> ``&amp;``, ``"`` -> ``&quot;``), so ``_attrs`` reverses
them on decode (``&quot;`` before ``&amp;`` -- reverse of encode order).

Known limitation: because string argument and response bodies are emitted raw,
a value that literally contains ``<|close|>argument<|sep|>`` or
``<|close|>response<|sep|>`` is indistinguishable from a real closing marker.
"""

import json
from collections.abc import Sequence
from functools import wraps
from typing import Any

import regex as re
from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import DelegatingParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager
from vllm.tool_parsers.structural_tag_registry import (
    SimplifiedToolChoice,
    _get_function_parameters,
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
    register_model_structural_tag,
)
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
)

from vllm_ascend.patch.platform.patch_kimi_k3_reasoning_parser import (
    KimiK3ReasoningParser,
)

_ORIGINAL_EXTRACT_TOOL_CALLS_ATTR = "_ascend_original_kimi_k3_extract_tool_calls"
_ORIGINAL_EXTRACT_TOOL_CALLS_STREAMING_ATTR = "_ascend_original_kimi_k3_extract_tool_calls_streaming"
_ORIGINAL_PARSE_DELTA_ATTR = "_ascend_original_kimi_k3_parse_delta"
_ORIGINAL_CHAT_FULL_ATTR = "_ascend_original_kimi_k3_chat_completion_full_generator"
_ORIGINAL_CHAT_STREAM_ATTR = "_ascend_original_kimi_k3_chat_completion_stream_generator"

_K3_OPEN = "<|open|>"
_K3_CLOSE = "<|close|>"
_K3_SEP = "<|sep|>"
_K3_END_OF_MSG = "<|end_of_msg|>"
_K3_THINK_CLOSE = f"{_K3_CLOSE}think{_K3_SEP}"
_K3_RESPONSE_OPEN = f"{_K3_OPEN}response{_K3_SEP}"
_K3_RESPONSE_CLOSE = f"{_K3_CLOSE}response{_K3_SEP}"
_K3_TOOLS_OPEN = f"{_K3_OPEN}tools{_K3_SEP}"
_K3_TOOLS_CLOSE = f"{_K3_CLOSE}tools{_K3_SEP}"
_K3_CALL_OPEN = f"{_K3_OPEN}call"
_K3_CALL_CLOSE = f"{_K3_CLOSE}call{_K3_SEP}"
_K3_ARG_OPEN = f"{_K3_OPEN}argument"
_K3_ARG_CLOSE = f"{_K3_CLOSE}argument{_K3_SEP}"
_K3_JSON_OPEN = f"{_K3_OPEN}json"
_K3_JSON_CLOSE = f"{_K3_CLOSE}json{_K3_SEP}"
_K3_MESSAGE_CLOSE = f"{_K3_CLOSE}message{_K3_SEP}"
_K3_JSON_TO_XTML_TYPE = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}
_K3_XTML_TYPES = tuple(dict.fromkeys(_K3_JSON_TO_XTML_TYPE.values()))


def _k3_escape_attr(value: str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _k3_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, dict):
        return "object"
    return "array"


def _k3_schema_types(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return list(_K3_XTML_TYPES)
    json_types = schema.get("type")
    if isinstance(json_types, str):
        json_types = [json_types]
    types = [
        xtml_type for json_type in json_types or [] if (xtml_type := _K3_JSON_TO_XTML_TYPE.get(json_type)) is not None
    ]
    if not types and "const" in schema:
        types = [_k3_value_type(schema["const"])]
    if not types and isinstance(schema.get("enum"), list):
        types = [_k3_value_type(value) for value in schema["enum"]]
    return list(dict.fromkeys(types)) or list(_K3_XTML_TYPES)


def _k3_string_content(schema: Any) -> Any:
    values = schema.get("enum") if isinstance(schema, dict) else None
    if values is None and isinstance(schema, dict) and "const" in schema:
        values = [schema["const"]]
    if (
        not isinstance(values, list)
        or not values
        or len(values) > 256
        or any(not isinstance(value, str) or "<|" in value for value in values)
    ):
        return AnyTextFormat(excludes=[_K3_ARG_CLOSE, _K3_CALL_CLOSE])
    formats: list[Any] = [ConstStringFormat(value=value) for value in values]
    return formats[0] if len(formats) == 1 else OrFormat(elements=formats)


def _k3_narrow_schema(schema: Any, xtml_type: str, root_defs: dict[str, Any]) -> dict[str, Any] | bool:
    if not isinstance(schema, dict):
        return True
    narrowed = dict(schema)
    json_types = schema.get("type")
    integer_only = json_types == "integer" or (
        isinstance(json_types, list) and "integer" in json_types and "number" not in json_types
    )
    narrowed["type"] = "integer" if xtml_type == "number" and integer_only else xtml_type
    for key, value in root_defs.items():
        narrowed.setdefault(key, value)
    return narrowed


def _k3_argument_tags(key: str, schema: Any, root_defs: dict[str, Any]) -> list[TagFormat]:
    tags = []
    for xtml_type in _k3_schema_types(schema):
        content = (
            _k3_string_content(schema)
            if xtml_type == "string"
            else JSONSchemaFormat(json_schema=_k3_narrow_schema(schema, xtml_type, root_defs))
        )
        tags.append(
            TagFormat(
                begin=f'{_K3_OPEN}argument key="{_k3_escape_attr(key)}" type="{xtml_type}"{_K3_SEP}',
                content=content,
                end=_K3_ARG_CLOSE,
            )
        )
    return tags


def _k3_permissive_argument_tag() -> TagFormat:
    key = RegexFormat(pattern=r'(?:[^<"&]|&(?:amp|quot);|<[^|])*')
    alternatives: list[Any] = []
    for xtml_type in _K3_XTML_TYPES:
        content = (
            AnyTextFormat(excludes=[_K3_ARG_CLOSE, _K3_CALL_CLOSE])
            if xtml_type == "string"
            else JSONSchemaFormat(json_schema=True)
        )
        alternatives.append(
            SequenceFormat(
                elements=[
                    key,
                    ConstStringFormat(value=f'" type="{xtml_type}"{_K3_SEP}'),
                    content,
                ]
            )
        )
    return TagFormat(
        begin=f'{_K3_OPEN}argument key="',
        content=OrFormat(elements=alternatives),
        end=_K3_ARG_CLOSE,
    )


def _k3_typed_arguments(parameters: dict[str, Any] | bool) -> Any:
    if parameters is False:
        return ConstStringFormat(value="")
    if not isinstance(parameters, dict):
        return StarFormat(content=_k3_permissive_argument_tag())
    props = parameters.get("properties")
    if not isinstance(props, dict) or not props:
        return StarFormat(content=_k3_permissive_argument_tag())

    root_defs = {
        defs_key: parameters[defs_key]
        for defs_key in ("$defs", "definitions")
        if isinstance(parameters.get(defs_key), dict)
    }
    tags: list[Any] = [tag for key, schema in props.items() for tag in _k3_argument_tags(key, schema, root_defs)]
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=tags)
    required = parameters.get("required")
    if isinstance(required, list) and required:
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


def _k3_raw_json_arguments(parameters: dict[str, Any] | bool) -> TagFormat:
    return TagFormat(
        begin=f'{_K3_JSON_OPEN} type="object"{_K3_SEP}',
        content=JSONSchemaFormat(json_schema=parameters),
        end=_K3_JSON_CLOSE,
    )


def _k3_call_tag(tool: ChatCompletionToolsParam) -> TagFormat:
    function = tool.function
    begin = f'{_K3_OPEN}call tool="{_k3_escape_attr(function.name)}" index="'
    return TagFormat(
        begin=begin,
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[1-9][0-9]*"),
                ConstStringFormat(value=f'"{_K3_SEP}'),
                OrFormat(
                    elements=[
                        _k3_typed_arguments(_get_function_parameters(function)),
                        _k3_raw_json_arguments(_get_function_parameters(function)),
                    ]
                ),
            ]
        ),
        end=_K3_CALL_CLOSE,
    )


def _k3_response_prefix(reasoning: bool) -> list[Any]:
    prefix: list[Any] = []
    if reasoning:
        prefix.extend(
            [
                TagFormat(
                    begin="",
                    content=AnyTextFormat(excludes=[_K3_THINK_CLOSE, _K3_END_OF_MSG]),
                    end=_K3_THINK_CLOSE,
                ),
                ConstStringFormat(value=_K3_RESPONSE_OPEN),
            ]
        )
    else:
        prefix.append(OptionalFormat(content=ConstStringFormat(value=_K3_RESPONSE_OPEN)))
    prefix.append(
        TagFormat(
            begin="",
            content=AnyTextFormat(excludes=[_K3_RESPONSE_CLOSE, _K3_TOOLS_OPEN, _K3_MESSAGE_CLOSE, _K3_END_OF_MSG]),
            end=_K3_RESPONSE_CLOSE,
        )
    )
    return prefix


def _k3_tools_channel(tools: list[ChatCompletionToolsParam]) -> TagFormat:
    return TagFormat(
        begin=_K3_TOOLS_OPEN,
        content=TagsWithSeparatorFormat(
            tags=[_k3_call_tag(tool) for tool in tools],
            separator="",
            at_least_one=True,
        ),
        end=_K3_TOOLS_CLOSE,
    )


@register_model_structural_tag("kimi_k3")
def get_kimi_k3_structural_tag(
    tools: list[ChatCompletionToolsParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    trailer = OptionalFormat(content=ConstStringFormat(value=_K3_MESSAGE_CLOSE))
    tools_part: Any | None
    if tool_choice == "auto" and not tools:
        tools_part = None
    elif tool_choice == "auto":
        tools_part = OptionalFormat(content=_k3_tools_channel(tools))
    elif tool_choice == "forced":
        tools_part = _k3_tools_channel(tools[:1])
    else:
        tools_part = _k3_tools_channel(tools)

    elements = _k3_response_prefix(reasoning)
    if tools_part is not None:
        elements.append(tools_part)
    elements.append(trailer)
    return StructuralTag(format=SequenceFormat(elements=elements))


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


def _safe_prefix_len(text: str, markers: Sequence[str]) -> int:
    positions = [position for marker in markers if (position := text.find(marker)) >= 0]
    if positions:
        return min(positions)
    overlap = max((_partial_tag_overlap(text, marker) for marker in markers), default=0)
    return len(text) - overlap


class KimiK3ToolParser(ToolParser):
    supports_required_and_named = False
    structural_tag_model = "kimi_k3"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self._attr_re = re.compile(r'\s+(?P<k>[\w]+)="(?P<v>[^"]*)"')
        self._reset()

    def get_structural_tag(self, request: ChatCompletionRequest):
        if request.tool_choice == "auto" and not any(
            getattr(tool.function, "strict", None) is True for tool in (request.tools or [])
        ):
            return None
        return get_model_structural_tag(
            model="kimi_k3",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        named = isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
        if (
            isinstance(request, ChatCompletionRequest)
            and request.tools
            and (request.tool_choice == "required" or named)
        ):
            structural_tag = self.get_structural_tag(request)
            if structural_tag is not None:
                encoded_tag = json.dumps(structural_tag.model_dump())
                if request.structured_outputs is None:
                    request.structured_outputs = StructuredOutputsParams(structural_tag=encoded_tag)
                else:
                    request.structured_outputs.structural_tag = encoded_tag

        structured_outputs = getattr(request, "structured_outputs", None)
        has_structural_tag = structured_outputs is not None and structured_outputs.structural_tag is not None
        if named and not has_structural_tag:
            # Without the XTML structural tag there is no way to force the
            # named call (the generic JSON guided-decoding path conflicts
            # with the XTML channel format).
            raise VLLMValidationError(
                "Named tool choice for Kimi K3 requires strict tool calling "
                "(VLLM_ENFORCE_STRICT_TOOL_CALLING) so the XTML structural "
                "tag can force the call. Otherwise use `tool_choice` set to "
                '"auto", "required", or "none".',
                parameter="tool_choice",
                value=request.tool_choice,
            )

        if request.tools and (request.tool_choice == "required" or named):
            request.skip_special_tokens = False
            if hasattr(request, "spaces_between_special_tokens"):
                request.spaces_between_special_tokens = False
            return request

        request = super().adjust_request(request)
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def _reset(self) -> None:
        self._buffer = ""
        self._mode = "idle"
        self._tool_calls: list[ToolCall] = []
        self._call_attrs = ""
        self._call_scan_pos = 0
        self._finished = False

    def _attrs(self, text: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        position = 0
        while position < len(text):
            match = self._attr_re.match(text, position)
            if match is None:
                raise ValueError("Invalid Kimi K3 XTML tag attributes.")
            attrs[match["k"]] = match["v"].replace("&quot;", '"').replace("&amp;", "&")
            position = match.end()
        return attrs

    def _parse_call_arguments(self, body: str) -> str:
        start = len(body) - len(body.lstrip())
        end = len(body.rstrip())
        payload = body[start:end]
        if payload.startswith(_K3_JSON_OPEN):
            separator = payload.find(_K3_SEP, len(_K3_JSON_OPEN))
            if separator < 0 or not payload.endswith(_K3_JSON_CLOSE):
                raise ValueError("Invalid Kimi K3 json argument block.")
            return payload[separator + len(_K3_SEP) : -len(_K3_JSON_CLOSE)]

        arguments: dict[str, Any] = {}
        position = 0
        while position < len(payload):
            while position < len(payload) and payload[position].isspace():
                position += 1
            if position == len(payload):
                break
            if not payload.startswith(_K3_ARG_OPEN, position):
                raise ValueError("Invalid Kimi K3 call body.")
            separator = payload.find(_K3_SEP, position + len(_K3_ARG_OPEN))
            if separator < 0:
                raise ValueError("Invalid Kimi K3 argument block.")
            close = payload.find(_K3_ARG_CLOSE, separator + len(_K3_SEP))
            if close < 0:
                raise ValueError("Invalid Kimi K3 argument block.")

            attrs = self._attrs(payload[position + len(_K3_ARG_OPEN) : separator])
            key = attrs.get("key", "")
            arg_type = attrs.get("type", "string")
            raw_value = payload[separator + len(_K3_SEP) : close]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
            position = close + len(_K3_ARG_CLOSE)

        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

    def _decode_call(self, attrs: str, body: str) -> ToolCall | None:
        call_attrs = self._attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        tool_index = call_attrs.get("index", "")
        if not tool_name:
            return None
        tool_call_id = tool_name
        if tool_index:
            try:
                tool_call_id = f"{tool_name}:{int(tool_index) - 1}"
            except ValueError:
                tool_call_id = f"{tool_name}:{tool_index}"
        # id uses the API-side zero-based call ordinal; XTML's message index
        # stays one-based when rendering tool result messages.
        return ToolCall(
            id=tool_call_id,
            type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=self._parse_call_arguments(body),
            ),
        )

    def _feed(self, chunk: str) -> tuple[str, list[ToolCall]]:
        if self._finished:
            raise ValueError("Kimi K3 parser received text after finish().")
        self._buffer += chunk
        content: list[str] = []
        new_calls: list[ToolCall] = []

        while self._buffer:
            if self._mode == "done":
                self._buffer = ""
                break

            if self._mode == "call":
                close = self._buffer.find(_K3_CALL_CLOSE, self._call_scan_pos)
                if close < 0:
                    self._call_scan_pos = max(
                        0,
                        len(self._buffer) - len(_K3_CALL_CLOSE) + 1,
                    )
                    break
                tool_call = self._decode_call(self._call_attrs, self._buffer[:close])
                self._buffer = self._buffer[close + len(_K3_CALL_CLOSE) :]
                self._mode = "tools"
                self._call_scan_pos = 0
                if tool_call is not None:
                    self._tool_calls.append(tool_call)
                    new_calls.append(tool_call)
                continue

            markers: Sequence[str]
            if self._mode == "idle":
                markers = (
                    _K3_RESPONSE_OPEN,
                    _K3_RESPONSE_CLOSE,
                    _K3_TOOLS_OPEN,
                    _K3_MESSAGE_CLOSE,
                    _K3_END_OF_MSG,
                )
                emit_text = True
            elif self._mode == "response":
                markers = (
                    _K3_RESPONSE_CLOSE,
                    _K3_TOOLS_OPEN,
                    _K3_MESSAGE_CLOSE,
                    _K3_END_OF_MSG,
                )
                emit_text = True
            elif self._mode == "epilogue":
                markers = (_K3_TOOLS_OPEN, _K3_MESSAGE_CLOSE, _K3_END_OF_MSG)
                emit_text = False
            else:
                markers = (_K3_CALL_OPEN, _K3_TOOLS_CLOSE, _K3_MESSAGE_CLOSE, _K3_END_OF_MSG)
                emit_text = False

            safe_len = _safe_prefix_len(self._buffer, markers)
            if safe_len:
                if emit_text:
                    content.append(self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
                continue

            if self._mode == "tools" and self._buffer.startswith(_K3_CALL_OPEN):
                separator = self._buffer.find(_K3_SEP, len(_K3_CALL_OPEN))
                if separator < 0:
                    break
                self._call_attrs = self._buffer[len(_K3_CALL_OPEN) : separator]
                self._attrs(self._call_attrs)
                self._buffer = self._buffer[separator + len(_K3_SEP) :]
                self._mode = "call"
                self._call_scan_pos = 0
                continue
            if self._buffer.startswith(_K3_RESPONSE_OPEN):
                self._buffer = self._buffer[len(_K3_RESPONSE_OPEN) :]
                self._mode = "response"
                continue
            if self._buffer.startswith(_K3_RESPONSE_CLOSE):
                self._buffer = self._buffer[len(_K3_RESPONSE_CLOSE) :]
                self._mode = "epilogue"
                continue
            if self._buffer.startswith(_K3_TOOLS_OPEN):
                self._buffer = self._buffer[len(_K3_TOOLS_OPEN) :]
                self._mode = "tools"
                continue
            if self._buffer.startswith(_K3_TOOLS_CLOSE):
                self._buffer = self._buffer[len(_K3_TOOLS_CLOSE) :]
                self._mode = "epilogue"
                continue
            if self._buffer.startswith(_K3_MESSAGE_CLOSE):
                self._buffer = self._buffer[len(_K3_MESSAGE_CLOSE) :]
                self._mode = "done"
                continue
            if self._buffer.startswith(_K3_END_OF_MSG):
                self._buffer = self._buffer[len(_K3_END_OF_MSG) :]
                self._mode = "done"
                continue
            break

        return "".join(content), new_calls

    def _finish(self) -> str:
        if self._finished:
            return ""
        if self._mode in ("idle", "response"):
            content = self._buffer
        elif self._mode == "call" or (self._mode == "tools" and self._buffer):
            raise ValueError("incomplete Kimi K3 tool call")
        else:
            content = ""
        self._buffer = ""
        self._mode = "idle"
        self._finished = True
        return content

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        del request
        self._reset()
        content, _ = self._feed(model_output)
        content += self._finish()
        return ExtractedToolCallInformation(
            tools_called=bool(self._tool_calls),
            tool_calls=self._tool_calls,
            content=content or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_text, current_text, previous_token_ids, current_token_ids, delta_token_ids, request
        content, calls = self._feed(delta_text)
        deltas = [
            DeltaToolCall(
                index=len(self._tool_calls) - len(calls) + i,
                id=tc.id,
                type="function",
                function=DeltaFunctionCall(name=tc.function.name, arguments=tc.function.arguments).model_dump(
                    exclude_none=True
                ),
            )
            for i, tc in enumerate(calls)
        ]
        return DeltaMessage(content=content or None, tool_calls=deltas) if content or deltas else None

    def finish_streaming(self) -> DeltaMessage | None:
        content = self._finish()
        return DeltaMessage(content=content) if content else None


if not hasattr(DelegatingParser, _ORIGINAL_EXTRACT_TOOL_CALLS_ATTR):
    setattr(
        DelegatingParser,
        _ORIGINAL_EXTRACT_TOOL_CALLS_ATTR,
        DelegatingParser._extract_tool_calls,
    )


@wraps(getattr(DelegatingParser, _ORIGINAL_EXTRACT_TOOL_CALLS_ATTR))
def _extract_tool_calls_with_kimi_k3(
    self: DelegatingParser,
    content: str | None,
    request: ChatCompletionRequest | ResponsesRequest,
    enable_auto_tools: bool = False,
) -> tuple[list[FunctionCall] | None, str | None]:
    tool_parser = self.tool_parser
    if not isinstance(tool_parser, KimiK3ToolParser):
        original = getattr(DelegatingParser, _ORIGINAL_EXTRACT_TOOL_CALLS_ATTR)
        return original(self, content, request, enable_auto_tools)

    if request.tool_choice == "none":
        extracted = tool_parser.extract_tool_calls(content or "", request)  # type: ignore[arg-type]
        return [], extracted.content

    named = isinstance(
        request.tool_choice,
        (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
    )
    forced = request.tool_choice == "required" or named
    automatic = enable_auto_tools and request.tool_choice in (None, "auto")
    if not forced and not automatic:
        return [], content

    extracted = tool_parser.extract_tool_calls(content or "", request)  # type: ignore[arg-type]
    if not extracted.tools_called:
        return None, extracted.content

    tool_calls = [
        FunctionCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        for tool_call in extracted.tool_calls
    ]
    remaining_content = extracted.content
    if remaining_content and not remaining_content.strip():
        remaining_content = None
    return tool_calls, remaining_content


DelegatingParser._extract_tool_calls = _extract_tool_calls_with_kimi_k3


if not hasattr(DelegatingParser, _ORIGINAL_EXTRACT_TOOL_CALLS_STREAMING_ATTR):
    setattr(
        DelegatingParser,
        _ORIGINAL_EXTRACT_TOOL_CALLS_STREAMING_ATTR,
        DelegatingParser._extract_tool_calls_streaming,
    )


@wraps(getattr(DelegatingParser, _ORIGINAL_EXTRACT_TOOL_CALLS_STREAMING_ATTR))
def _extract_tool_calls_streaming_with_kimi_k3(
    self: DelegatingParser,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
    request: ChatCompletionRequest | ResponsesRequest,
    tool_call_idx: int | None = None,
    tool_call_id_type: str = "random",
    function_name_returned: bool = False,
) -> tuple[DeltaMessage | None, bool]:
    tool_parser = self.tool_parser
    if not isinstance(tool_parser, KimiK3ToolParser) or request.tool_choice != "none":
        original = getattr(
            DelegatingParser,
            _ORIGINAL_EXTRACT_TOOL_CALLS_STREAMING_ATTR,
        )
        return original(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
            tool_call_idx,
            tool_call_id_type,
            function_name_returned,
        )

    delta = tool_parser.extract_tool_calls_streaming(
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,  # type: ignore[arg-type]
    )
    if delta is not None:
        delta.tool_calls = []
    return delta, False


DelegatingParser._extract_tool_calls_streaming = _extract_tool_calls_streaming_with_kimi_k3


if not hasattr(DelegatingParser, _ORIGINAL_PARSE_DELTA_ATTR):
    setattr(DelegatingParser, _ORIGINAL_PARSE_DELTA_ATTR, DelegatingParser.parse_delta)


@wraps(getattr(DelegatingParser, _ORIGINAL_PARSE_DELTA_ATTR))
def _parse_delta_with_kimi_k3_finish(
    self: DelegatingParser,
    delta_text: str,
    delta_token_ids: list[int],
    request: ChatCompletionRequest | ResponsesRequest,
    prompt_token_ids: list[int] | None = None,
    *,
    finished: bool,
) -> DeltaMessage | None:
    original = getattr(DelegatingParser, _ORIGINAL_PARSE_DELTA_ATTR)
    delta = original(
        self,
        delta_text,
        delta_token_ids,
        request,
        prompt_token_ids,
        finished=finished,
    )
    if not finished:
        return delta

    reasoning_parser = self.reasoning_parser
    if isinstance(reasoning_parser, KimiK3ReasoningParser) and not self._stream_state.reasoning_ended:
        tail = reasoning_parser.finish_streaming(self._stream_state.previous_text)
        if tail is not None:
            if delta is None:
                delta = tail
            elif tail.reasoning:
                delta.reasoning = (delta.reasoning or "") + tail.reasoning

    tool_parser = self.tool_parser
    if not isinstance(tool_parser, KimiK3ToolParser) or not self._stream_state.tool_call_text_started:
        return delta

    tail = tool_parser.finish_streaming()
    if tail is None:
        return delta
    if delta is None:
        return tail
    if tail.content:
        delta.content = (delta.content or "") + tail.content
    return delta


DelegatingParser.parse_delta = _parse_delta_with_kimi_k3_finish


def _is_kimi_k3_parser(parser: Any) -> bool:
    return isinstance(parser, DelegatingParser) and isinstance(parser.tool_parser, KimiK3ToolParser)


def _is_remote_decode_request(request: ChatCompletionRequest) -> bool:
    params = request.kv_transfer_params
    return bool(params and params.get("do_remote_decode") is True)


async def _capture_finish_reasons(result_generator, finish_reasons: dict[int, str]):
    async for result in result_generator:
        for output in result.outputs:
            if output.finish_reason is not None:
                finish_reasons[output.index] = output.finish_reason
        yield result


def _restore_engine_finish_reason(data: str, finish_reasons: dict[int, str]) -> str:
    if not finish_reasons or '"finish_reason":"tool_calls"' not in data:
        return data
    if not data.startswith("data: ") or not data.endswith("\n\n"):
        return data

    payload = data[len("data: ") : -len("\n\n")]
    if payload == "[DONE]":
        return data
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return data

    changed = False
    for choice in chunk.get("choices") or []:
        engine_reason = finish_reasons.get(choice.get("index"))
        if choice.get("finish_reason") == "tool_calls" and engine_reason not in (None, "stop"):
            choice["finish_reason"] = engine_reason
            changed = True
    if not changed:
        return data
    return f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _split_terminal_tool_delta(data: str) -> list[str]:
    if '"tool_calls"' not in data or '"finish_reason"' not in data:
        return [data]
    if not data.startswith("data: ") or not data.endswith("\n\n"):
        return [data]
    try:
        payload = json.loads(data[len("data: ") : -len("\n\n")])
    except json.JSONDecodeError:
        return [data]

    data_choices = []
    terminal_choices = []
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        if choice.get("finish_reason") is None or not delta.get("tool_calls"):
            data_choices.append(choice)
            continue

        data_choice = dict(choice)
        data_choice["finish_reason"] = None
        data_choice.pop("stop_reason", None)
        data_choices.append(data_choice)

        terminal_choice = dict(choice)
        terminal_choice["delta"] = {}
        terminal_choice.pop("logprobs", None)
        terminal_choice.pop("token_ids", None)
        if terminal_choice.get("finish_reason") == "tool_calls":
            terminal_choice.pop("stop_reason", None)
        terminal_choices.append(terminal_choice)

    if not terminal_choices:
        return [data]

    data_payload = dict(payload)
    data_payload["choices"] = data_choices
    terminal_payload = dict(payload)
    terminal_payload["choices"] = terminal_choices
    data_json = json.dumps(data_payload, ensure_ascii=False, separators=(",", ":"))
    terminal_json = json.dumps(terminal_payload, ensure_ascii=False, separators=(",", ":"))
    return [f"data: {data_json}\n\n", f"data: {terminal_json}\n\n"]


async def _chat_completion_full_generator_with_kimi_k3(
    self,
    request,
    result_generator,
    request_id,
    model_name,
    conversation,
    tokenizer,
    request_metadata,
    parser=None,
):
    original = getattr(self, _ORIGINAL_CHAT_FULL_ATTR)
    if not _is_kimi_k3_parser(parser):
        return await original(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser,
        )
    if _is_remote_decode_request(request):
        return await original(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            None,
        )

    finish_reasons: dict[int, str] = {}
    parsed_contents: list[str | None] = []
    original_parse = parser.parse

    def parse_and_capture(*args, **kwargs):
        parsed = original_parse(*args, **kwargs)
        parsed_contents.append(parsed[1])
        return parsed

    parser.parse = parse_and_capture
    try:
        response = await original(
            request,
            _capture_finish_reasons(result_generator, finish_reasons),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser,
        )
    finally:
        del parser.parse

    if isinstance(response, ChatCompletionResponse):
        for choice in response.choices:
            engine_reason = finish_reasons.get(choice.index)
            if choice.finish_reason == "tool_calls" and engine_reason not in (None, "stop"):
                choice.finish_reason = engine_reason

        named = isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
        if request.tool_choice == "required" or named:
            for choice, content in zip(response.choices, parsed_contents, strict=False):
                choice.message.content = content or ""
    return response


async def _chat_completion_stream_generator_with_kimi_k3(
    self,
    request,
    result_generator,
    *args,
    **kwargs,
):
    original = getattr(self, _ORIGINAL_CHAT_STREAM_ATTR)
    parser_cls = self.parser_cls
    if parser_cls is None or getattr(parser_cls, "tool_parser_cls", None) is not KimiK3ToolParser:
        async for data in original(request, result_generator, *args, **kwargs):
            yield data
        return

    finish_reasons: dict[int, str] = {}
    async for data in original(
        request,
        _capture_finish_reasons(result_generator, finish_reasons),
        *args,
        **kwargs,
    ):
        restored = _restore_engine_finish_reason(data, finish_reasons)
        for chunk in _split_terminal_tool_delta(restored):
            yield chunk


if not hasattr(OpenAIServingChat, _ORIGINAL_CHAT_FULL_ATTR):
    setattr(
        OpenAIServingChat,
        _ORIGINAL_CHAT_FULL_ATTR,
        OpenAIServingChat.chat_completion_full_generator,
    )
    setattr(
        OpenAIServingChat,
        _ORIGINAL_CHAT_STREAM_ATTR,
        OpenAIServingChat.chat_completion_stream_generator,
    )
    OpenAIServingChat.chat_completion_full_generator = _chat_completion_full_generator_with_kimi_k3
    OpenAIServingChat.chat_completion_stream_generator = _chat_completion_stream_generator_with_kimi_k3


ToolParserManager.register_module(
    name="kimi_k3",
    module=KimiK3ToolParser,
    force=True,
)
