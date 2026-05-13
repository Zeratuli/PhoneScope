from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address)

service_enabled = True


def get_service_status() -> bool:
    return service_enabled


def set_service_status(enabled: bool) -> None:
    global service_enabled
    service_enabled = enabled


class ServiceToggleMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        exempt_paths = ["/api/v1/health", "/api/v1/admin/toggle", "/docs", "/openapi.json"]
        if not service_enabled and not any(path.startswith(p) for p in exempt_paths):
            return JSONResponse(
                status_code=503,
                content={"detail": "服务已暂停，请联系管理员"},
            )
        return await call_next(request)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    return JSONResponse(
        status_code=429,
        content={"detail": f"请求过于频繁，请稍后再试。限制: {exc.detail}"},
    )
