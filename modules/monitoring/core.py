from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
import psutil
import time
import logging
import structlog
from prometheus_client import generate_latest, REGISTRY, Counter, Gauge, Histogram
from ..middleware.auth import protect, Permission, Role

# ========== Monitoring Setup ==========
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP Request Duration',
    ['method', 'endpoint']
)
ERROR_COUNT = Counter(
    'http_exceptions_total',
    'Total HTTP Exceptions',
    ['method', 'endpoint', 'exception']
)
SYSTEM_CPU = Gauge('system_cpu_usage', 'System CPU Usage (%)')
SYSTEM_MEM = Gauge('system_memory_usage', 'System Memory Usage (%)')

# Structured Logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)
logger = structlog.get_logger()

# ========== Health Check Service ==========
class HealthCheck:
    def __init__(self):
        self.services = {
            'database': self.check_database,
            'cache': self.check_cache,
            'external_api': self.check_external_api
        }
    
    async def check_database(self):
        try:
            # Implement actual database ping
            return {'status': 'ok', 'latency': 0.02}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def check_cache(self):
        # Implement cache connection check
        return {'status': 'ok'}
    
    async def check_external_api(self):
        # Implement external API health check
        return {'status': 'ok'}
    
    async def system_status(self):
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }

health_check = HealthCheck()

# ========== Monitoring Endpoints ==========
@router.get("/health", dependencies=[Depends(protect([Permission.CONFIGURE_SYSTEM]))])
async def health():
    services = {}
    for name, check in health_check.services.items():
        services[name] = await check()
    
    system = await health_check.system_status()
    return {
        'status': 'ok' if all(s['status'] == 'ok' for s in services.values()) else 'degraded',
        'services': services,
        'system': system
    }

@router.get("/metrics")
async def metrics():
    # Update system metrics
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEM.set(psutil.virtual_memory().percent)
    
    return generate_latest(REGISTRY)

# ========== Monitoring Middleware ==========
class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        response = None
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Structured logging
            logger.info(
                "request_processed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration=duration,
                client=request.client.host if request.client else None
            )
            
        except Exception as e:
            ERROR_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                exception=type(e).__name__
            ).inc()
            
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                traceback=logging.getExceptionMessage(e)
            )
            raise
        
        return response

# ========== Usage ==========
# Add to FastAPI app:
# app.add_middleware(MonitoringMiddleware)
# app.include_router(monitoring_router)

# ========== Alert Integration ==========
class PerformanceAlert:
    def __init__(self):
        self.thresholds = {
            'cpu': 90,
            'memory': 85,
            'disk': 80
        }
    
    async def check_thresholds(self):
        system = await health_check.system_status()
        alerts = []
        for metric, value in system.items():
            if value > self.thresholds[metric]:
                alerts.append(f"{metric} usage at {value}%")
        return alerts 