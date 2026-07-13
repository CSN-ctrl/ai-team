from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send


class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for SPA routing."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except Exception:
            return await super().get_response("index.html", scope)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            response = Response(status_code=404)
            await response(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
