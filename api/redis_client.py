from functools import lru_cache
from redis import Redis, ConnectionPool

from .config import settings


# ---------------------------------------------------------
# Redis Client Provider
# ---------------------------------------------------------
@lru_cache()
def get_redis_client() -> Redis:
    """
    Creates and returns a cached Redis client instance.
    Uses lru_cache to ensure a singleton pattern for the client.

    Returns:
        Redis: Configured Redis client instance ready for operations.
    """
    # Get Redis configuration from settings
    redis_host = settings.redis_host
    redis_port = settings.redis_port
    db = getattr(settings, "redis_db", 0)

    # Create connection pool
    connection_pool = ConnectionPool(host=redis_host, port=redis_port, db=db, decode_responses=True, max_connections=10)

    # Create Redis client instance
    redis_client = Redis(connection_pool=connection_pool)

    return redis_client
