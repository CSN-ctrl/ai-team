"""OpenAI-compatible chat completions proxy — /v1/chat/completions.

This endpoint lets tools such as Open WebUI connect to OpenClaw as an
OpenAI-compatible backend.  Requests are forwarded to the NVIDIA NIM API
via the ``NIMClient`` instance stored in ``app.state.nim_client``.

Mode routing (Ask / Plan / Code / Auto) is driven by the ``X-AI-Mode``
header or, in Auto mode, by an intent classifier.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm.client import NIMClient
from app.llm.prompts import ASK_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from app.models.mode import ConversationManager, Mode

from app.routers.chat_mode import (
    classify_and_route,
    get_or_create_conv_id,
    make_plan_complete_callback,
    prepare_messages,
    process_plan_text,
    resolve_mode,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

_DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = _DEFAULT_MODEL
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    extra: dict[str, Any] = {}


def _client(request: Request) -> NIMClient:
    client = getattr(request.app.state, "nim_client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="NIM client not available — check API key configuration",
        )
    return client


async def _stream_chunks(
    client: NIMClient,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    tools: Optional[list[dict]] = None,
    on_complete: Optional[Callable[[str], Any]] = None,
) -> AsyncGenerator[str, None]:
    """Yields SSE ``data: ...`` lines for an OpenAI-compatible stream.

    When *on_complete* is provided, the full response text is passed to
    it after the terminal chunk is yielded (fire-and-forget).
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    preamble = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(preamble, ensure_ascii=False)}\n\n"

    full_content = ""
    try:
        async for delta in client.chat_completion_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        ):
            full_content += delta
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    except Exception as exc:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    terminal = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(terminal, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    if on_complete is not None:
        try:
            result = on_complete(full_content)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.exception("Post-stream callback failed")


@router.get("/models")
async def list_models(request: Request) -> dict:
    """OpenAI-compatible model list — /v1/models."""
    models: list[dict] = []
    router_instance = getattr(request.app.state, "router", None)
    if router_instance is not None:
        try:
            raw = await router_instance.check_health()
            models = [
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nvidia",
                }
                for name, status in raw.items()
                if status
            ]
        except Exception:
            pass

    return {
        "object": "list",
        "data": models,
    }


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    """OpenAI-compatible chat completions proxy with mode routing.

    Mode is driven by the ``X-AI-Mode`` header (``ask`` | ``plan`` |
    ``code`` | ``auto``).  In ``auto`` mode an intent classifier chooses
    the route.  Conversation state is tracked via ``X-Conversation-Id``.

    Supports both streaming and non-streaming responses.
    """
    client = _client(request)
    messages = [m.dict() for m in body.messages]
    conv_id = get_or_create_conv_id(request)

    mode, _ = resolve_mode(request)

    if mode == Mode.AUTO:
        route = await classify_and_route(request, messages)
        mode = route.mode

    conv_manager: ConversationManager = getattr(
        request.app.state, "conv_manager", None
    )
    if conv_manager is not None:
        conv_manager.update_mode(conv_id, mode)

    if mode == Mode.ASK:
        prep_messages = prepare_messages(messages, ASK_SYSTEM_PROMPT)
        tools: Optional[list[dict]] = None
    elif mode == Mode.PLAN:
        prep_messages = prepare_messages(messages, PLAN_SYSTEM_PROMPT)
        tools = None
    else:
        prep_messages = messages
        tools = body.tools

    if body.stream:
        if mode == Mode.PLAN:
            on_complete = make_plan_complete_callback(request, conv_id)
        else:
            on_complete = None

        return StreamingResponse(
            _stream_chunks(
                client=client,
                model=body.model,
                messages=prep_messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                tools=tools,
                on_complete=on_complete,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await client.chat_completion(
            model=body.model,
            messages=prep_messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            tools=tools,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream model request failed: {exc}",
        )

    if mode == Mode.PLAN:
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if content:
            task_ids = await process_plan_text(request, content, conv_id)
            response["plan_meta"] = {
                "task_ids": task_ids,
                "conv_id": conv_id,
            }

    return response
