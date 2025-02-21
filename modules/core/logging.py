from loguru import logger
import sentry_sdk

def setup_logging():
    logger.add("logs/app.log", rotation="500 MB")
    
    sentry_sdk.init(
        dsn="your-sentry-dsn",
        traces_sample_rate=1.0,
    ) 