from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.routing import APIRoute
from fastapi import status
from typing import Callable, Dict, List, Optional
from datetime import datetime, timedelta
import re
import json
from ..middleware.auth import protect, Permission

# ========== Rate Limiting ==========
class RateLimiter:
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.access_records: Dict[str, List[datetime]] = {}

    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        now = datetime.now()
        
        # Cleanup old records
        self.access_records[client_ip] = [
            t for t in self.access_records.get(client_ip, []) 
            if now - t < timedelta(seconds=self.window)
        ]
        
        if len(self.access_records[client_ip]) >= self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests} requests per {self.window} seconds"
            )
            
        self.access_records.setdefault(client_ip, []).append(now)
        return True

# ========== Request Validation ==========
class RequestValidator:
    def __init__(self):
        self.schema_registry: Dict[str, dict] = {}
        
    def add_schema(self, endpoint: str, method: str, schema: dict):
        key = f"{method.upper()}:{endpoint}"
        self.schema_registry[key] = schema
        
    async def validate_request(self, request: Request):
        key = f"{request.method}:{request.url.path}"
        schema = self.schema_registry.get(key)
        
        if schema:
            try:
                body = await request.json()
                self._validate_body(body, schema)
            except json.JSONDecodeError:
                self._validate_headers(request.headers, schema)
                self._validate_query_params(request.query_params, schema)
                
    def _validate_body(self, body: dict, schema: dict):
        # Implement JSON Schema validation
        pass
        
    def _validate_headers(self, headers: dict, schema: dict):
        required_headers = schema.get("headers", [])
        for header in required_headers:
            if header not in headers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required header: {header}"
                )
                
    def _validate_query_params(self, params: dict, schema: dict):
        required_params = schema.get("query_params", [])
        for param in required_params:
            if param not in params:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required query parameter: {param}"
                )

# ========== API Versioning ==========
class VersionedAPIRoute(APIRoute):
    def __init__(self, *args, **kwargs):
        self.version = kwargs.pop("version", "v1")
        super().__init__(*args, **kwargs)

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def versioned_route_handler(request: Request):
            request.state.api_version = self.version
            return await original_route_handler(request)

        return versioned_route_handler

class APIVersionMiddleware:
    def __init__(self, app, default_version: str = "v1"):
        self.app = app
        self.default_version = default_version

    async def __call__(self, request: Request, call_next):
        version = self.get_version_from_request(request)
        request.state.api_version = version
        response = await call_next(request)
        response.headers["X-API-Version"] = version
        return response
        
    def get_version_from_request(self, request: Request) -> str:
        # Check header first
        version_header = request.headers.get("X-API-Version")
        if version_header and re.match(r"^v\d+$", version_header):
            return version_header
            
        # Check URL path
        path_version = request.url.path.split("/")[1]
        if re.match(r"^v\d+$", path_version):
            return path_version
            
        return self.default_version

# ========== Gateway Setup ==========
gateway = APIRouter(
    route_class=VersionedAPIRoute,
    dependencies=[Depends(protect([Permission.VIEW_ANALYTICS]))]
)

def create_gateway_middleware():
    return [
        Middleware(TrustedHostMiddleware, allowed_hosts=["*"]),
        Middleware(GZipMiddleware, minimum_size=1024),
        Middleware(APIVersionMiddleware),
    ]

# ========== Usage Example ==========
@gateway.get("/trades", version="v2")
async def get_trades():
    return {"version": "v2", "data": [...]}

# Add validation schema
validator = RequestValidator()
validator.add_schema(
    endpoint="/trades",
    method="GET",
    schema={
        "headers": ["X-Trading-Session"],
        "query_params": ["start_date", "end_date"],
        "body": {
            "type": "object",
            "properties": {
                "filter": {"type": "string"}
            }
        }
    }
)

# Apply rate limiting
limiter = RateLimiter(requests=100, window=60)

# Add middleware to FastAPI app:
# app = FastAPI(middleware=create_gateway_middleware())
# app.include_router(gateway) 