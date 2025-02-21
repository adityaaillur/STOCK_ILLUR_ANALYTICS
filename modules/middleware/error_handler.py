from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException as http_exc:
            return JSONResponse(
                status_code=http_exc.status_code,
                content={"detail": http_exc.detail}
            )
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"}
            ) 