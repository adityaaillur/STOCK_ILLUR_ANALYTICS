import redis
from loguru import logger
from datetime import timedelta
from typing import Optional, Any
import pickle

class RedisCache:
    """Redis-based caching layer"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False
        )
        logger.info("Redis cache initialized")
        
    def set(self, 
           key: str, 
           value: Any, 
           ttl: Optional[timedelta] = None) -> bool:
        """Set a cache value with optional TTL"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                return self.redis.setex(key, int(ttl.total_seconds()), serialized)
            else:
                return self.redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
            
    def get(self, key: str) -> Optional[Any]:
        """Get a cached value"""
        try:
            serialized = self.redis.get(key)
            if serialized:
                return pickle.loads(serialized)
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
            
    def delete(self, key: str) -> bool:
        """Delete a cache key"""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """Check if a cache key exists"""
        try:
            return bool(self.redis.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
            
    def clear(self) -> bool:
        """Clear all cache keys"""
        try:
            return bool(self.redis.flushdb())
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False 