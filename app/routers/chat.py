"""OpenAI-compatible chat completions proxy — /v1/chat/completions.

This endpoint lets tools such as Open WebUI connect to OpenClaw as an
OpenAI-compatible backend.  Requests are forwarded to the NVIDIA NIM API
via the ``NIMClient`` instance stored in ``app.state.nim_client``.
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm.client import NIMClient

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
    # Allow extra fields for future / proxy compatibility
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
) -> AsyncGenerator[str, None]:
    """Yields SSE ``data: ...`` lines for an OpenAI-compatible stream."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    # Role preamble chunk
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
        # Emit an error chunk so the client isn't left hanging
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

    # Terminal chunk
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
    """OpenAI-compatible chat completions proxy.

    Accepts OpenAI-format requests and routes them through the NIM client.
    Supports both streaming and non-streaming responses.
    """
    client = _client(request)
    messages = [m.dict() for m in body.messages]

    if body.stream:
        return StreamingResponse(
            _stream_chunks(
                client=client,
                model=body.model,
                messages=messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: pass through to NIMClient
    try:
        response = await client.chat_completion(
            model=body.model,
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream model request failed: {exc}",
        )
