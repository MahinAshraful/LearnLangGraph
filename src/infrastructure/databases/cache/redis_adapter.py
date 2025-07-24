import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, List, Dict
import logging

from .base import CacheAdapter
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisAdapter(CacheAdapter):
    """Redis cache adapter"""

    def __init__(self,
                 redis_url: Optional[str] = None,
                 key_prefix: str = "restaurant_rec",
                 serializer: str = "json"):  # "json" or "pickle"

        super().__init__(key_prefix)
        self.settings = get_settings()
        self.redis_url = redis_url or self.settings.database.redis_url
        self.serializer = serializer
        self.redis_client = None

        # Connection settings
        self.max_connections = 20
        self.socket_timeout = 30
        self.socket_connect_timeout = 30

    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=True if self.serializer == "json" else False
            )

            # Test connection
            await self.redis_client.ping()

            self.is_connected = True
            logger.info("Connected to Redis cache")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

        self.is_connected = False
        logger.info("Disconnected from Redis cache")

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            cache_key = self._make_key(key)
            value = await self.redis_client.get(cache_key)

            if value is None:
                return None

            return self._deserialize(value)

        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL"""
        try:
            cache_key = self._make_key(key)
            serialized_value = self._serialize(value)

            if ttl:
                await self.redis_client.setex(cache_key, ttl, serialized_value)
            else:
                await self.redis_client.set(cache_key, serialized_value)

            return True

        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.delete(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.exists(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.expire(cache_key, ttl)
            return result

        except Exception as e:
            logger.error(f"Redis expire failed for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            full_pattern = self._make_key(pattern)
            keys = await self.redis_client.keys(full_pattern)

            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Redis clear pattern failed for pattern {pattern}: {e}")
            return 0

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.redis_client.mget(cache_keys)

            result = {}
            for i, (original_key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    result[original_key] = self._deserialize(value)

            return result

        except Exception as e:
            logger.error(f"Redis get_many failed: {e}")
            return {}

    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs"""
        try:
            # Prepare data for Redis
            cache_mapping = {}
            for key, value in mapping.items():
                cache_key = self._make_key(key)
                cache_mapping[cache_key] = self._serialize(value)

            # Use pipeline for efficiency
            async with self.redis_client.pipeline() as pipe:
                await pipe.mset(cache_mapping)

                # Set TTL for all keys if specified
                if ttl:
                    for cache_key in cache_mapping.keys():
                        await pipe.expire(cache_key, ttl)

                await pipe.execute()

            return True

        except Exception as e:
            logger.error(f"Redis set_many failed: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.incrby(cache_key, amount)
            return result

        except Exception as e:
            logger.error(f"Redis increment failed for key {key}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            info = await self.redis_client.info()

            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "redis_version": info.get("redis_version", "unknown")
            }

        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        if self.serializer == "json":
            try:
                return json.dumps(value, default=str)
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects
                return pickle.dumps(value).hex()
        else:
            return pickle.dumps(value)

    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage"""
        if self.serializer == "json":
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Try pickle fallback
                try:
                    return pickle.loads(bytes.fromhex(value))
                except:
                    return value
        else:
            return pickle.loads(value)