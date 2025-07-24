import asyncio
import time
from typing import Any, Optional, List, Dict
from dataclasses import dataclass
import logging
import threading

from .base import CacheAdapter

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """In-memory cache entry"""
    value: Any
    expires_at: Optional[float] = None
    created_at: float = None
    access_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def access(self):
        """Record access to this entry"""
        self.access_count += 1


class MemoryAdapter(CacheAdapter):
    """In-memory cache adapter for development and testing"""

    def __init__(self,
                 key_prefix: str = "restaurant_rec",
                 max_size: int = 10000,
                 cleanup_interval: int = 300):  # 5 minutes

        super().__init__(key_prefix)
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        # Thread-safe storage
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }

        # Cleanup task
        self._cleanup_task = None

    async def connect(self) -> bool:
        """Connect to memory cache (always succeeds)"""
        self.is_connected = True

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info("Connected to Memory cache")
        return True

    async def disconnect(self):
        """Disconnect from memory cache"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        with self._lock:
            self._cache.clear()

        self.is_connected = False
        logger.info("Disconnected from Memory cache")

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        cache_key = self._make_key(key)

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[cache_key]
                self._stats["misses"] += 1
                return None

            entry.access()
            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL"""
        cache_key = self._make_key(key)

        # Calculate expiration time
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        entry = CacheEntry(
            value=value,
            expires_at=expires_at
        )

        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_entries()

            self._cache[cache_key] = entry
            self._stats["sets"] += 1

        return True

    async def delete(self, key: str) -> bool:
        """Delete key"""
        cache_key = self._make_key(key)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats["deletes"] += 1
                return True

            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        cache_key = self._make_key(key)

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return False

            if entry.is_expired:
                del self._cache[cache_key]
                return False

            return True

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        cache_key = self._make_key(key)

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return False

            entry.expires_at = time.time() + ttl
            return True

    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        full_pattern = self._make_key(pattern)

        # Simple pattern matching (supports * wildcard)
        if "*" in full_pattern:
            prefix = full_pattern.split("*")[0]
            suffix = full_pattern.split("*")[-1] if full_pattern.endswith("*") else ""

            with self._lock:
                keys_to_delete = []
                for cache_key in self._cache.keys():
                    if cache_key.startswith(prefix) and cache_key.endswith(suffix):
                        keys_to_delete.append(cache_key)

                for cache_key in keys_to_delete:
                    del self._cache[cache_key]

                self._stats["deletes"] += len(keys_to_delete)
                return len(keys_to_delete)
        else:
            # Exact match
            return 1 if await self.delete(pattern) else 0

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        result = {}

        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value

        return result

    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs"""
        try:
            for key, value in mapping.items():
                await self.set(key, value, ttl)
            return True
        except Exception as e:
            logger.error(f"Memory set_many failed: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        cache_key = self._make_key(key)

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None or entry.is_expired:
                # Create new counter
                new_entry = CacheEntry(value=amount)
                self._cache[cache_key] = new_entry
                return amount

            if isinstance(entry.value, (int, float)):
                entry.value += amount
                entry.access()
                return int(entry.value)
            else:
                # Invalid type for increment
                raise ValueError(f"Cannot increment non-numeric value: {type(entry.value)}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate_percent": round(hit_rate, 2),
                **self._stats
            }

    def _evict_entries(self):
        """Evict entries when cache is full (LRU-style)"""
        if len(self._cache) < self.max_size:
            return

        # Remove expired entries first
        expired_keys = []
        for cache_key, entry in self._cache.items():
            if entry.is_expired:
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            del self._cache[cache_key]

        # If still too many, remove least accessed entries
        if len(self._cache) >= self.max_size:
            # Sort by access count (ascending) and creation time
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda item: (item[1].access_count, item[1].created_at)
            )

            # Remove 10% of entries
            num_to_remove = max(1, len(self._cache) // 10)
            for i in range(num_to_remove):
                if i < len(sorted_entries):
                    cache_key = sorted_entries[i][0]
                    del self._cache[cache_key]
                    self._stats["evictions"] += 1

    async def _periodic_cleanup(self):
        """Periodically clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                with self._lock:
                    expired_keys = []
                    for cache_key, entry in self._cache.items():
                        if entry.is_expired:
                            expired_keys.append(cache_key)

                    for cache_key in expired_keys:
                        del self._cache[cache_key]

                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


# Factory function for creating cache adapters
def create_cache_adapter(cache_type: str = "memory", **kwargs) -> CacheAdapter:
    """Factory function to create cache adapters"""

    if cache_type.lower() == "redis":
        return RedisAdapter(**kwargs)
    elif cache_type.lower() == "memory":
        return MemoryAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


# Cache manager for handling multiple cache layers
class CacheManager:
    """Multi-layer cache manager"""

    def __init__(self, primary: CacheAdapter, secondary: Optional[CacheAdapter] = None):
        self.primary = primary
        self.secondary = secondary

    async def connect(self) -> bool:
        """Connect to all cache layers"""
        primary_ok = await self.primary.connect()
        secondary_ok = True

        if self.secondary:
            secondary_ok = await self.secondary.connect()

        return primary_ok and secondary_ok

    async def disconnect(self):
        """Disconnect from all cache layers"""
        await self.primary.disconnect()
        if self.secondary:
            await self.secondary.disconnect()

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback"""
        # Try primary cache first
        value = await self.primary.get(key)

        if value is not None:
            return value

        # Try secondary cache
        if self.secondary:
            value = await self.secondary.get(key)

            # If found in secondary, promote to primary
            if value is not None:
                await self.primary.set(key, value)
                return value

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in all cache layers"""
        primary_ok = await self.primary.set(key, value, ttl)
        secondary_ok = True

        if self.secondary:
            secondary_ok = await self.secondary.set(key, value, ttl)

        return primary_ok and secondary_ok

    async def delete(self, key: str) -> bool:
        """Delete from all cache layers"""
        primary_ok = await self.primary.delete(key)
        secondary_ok = True

        if self.secondary:
            secondary_ok = await self.secondary.delete(key)

        return primary_ok or secondary_ok  # Success if deleted from any layer

    async def health_check(self) -> Dict[str, Any]:
        """Health check for all cache layers"""
        primary_health = await self.primary.health_check()

        result = {
            "primary": primary_health,
            "overall_status": primary_health["status"]
        }

        if self.secondary:
            secondary_health = await self.secondary.health_check()
            result["secondary"] = secondary_health

            # Overall status is healthy if primary is healthy
            result["overall_status"] = primary_health["status"]

        return result