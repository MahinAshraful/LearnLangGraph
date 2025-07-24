from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class CacheAdapter(ABC):
    """Abstract base class for cache adapters"""

    def __init__(self, key_prefix: str = "restaurant_rec"):
        self.key_prefix = key_prefix
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the cache backend"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the cache backend"""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL (seconds)"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        pass

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        pass

    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        pass

    @abstractmethod
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs"""
        pass

    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key"""
        return f"{self.key_prefix}:{key}"

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        try:
            # Test basic operations
            test_key = self._make_key("health_check")
            await self.set(test_key, "ok", ttl=10)
            value = await self.get(test_key)
            await self.delete(test_key)

            stats = await self.get_stats()

            return {
                "status": "healthy" if value == "ok" else "unhealthy",
                "connected": self.is_connected,
                **stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }