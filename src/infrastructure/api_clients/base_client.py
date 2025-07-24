import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import aiohttp
import json

from ...config.settings import get_settings
from ...domain.exceptions.recommendation_errors import APIError, RateLimitError, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ClientStatus(str, Enum):
    """API client status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"


@dataclass
class APIResponse(Generic[T]):
    """Standardized API response wrapper"""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: float = 0.0
    cached: bool = False
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None

    @classmethod
    def success_response(cls, data: T, response_time_ms: float = 0.0, cached: bool = False) -> 'APIResponse[T]':
        return cls(success=True, data=data, response_time_ms=response_time_ms, cached=cached)

    @classmethod
    def error_response(cls, error: str, status_code: Optional[int] = None) -> 'APIResponse[T]':
        return cls(success=False, error=error, status_code=status_code)


@dataclass
class RateLimiter:
    """Simple rate limiter for API calls"""

    max_requests: int
    time_window_seconds: int
    requests: List[float] = field(default_factory=list)

    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limit"""
        now = time.time()
        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests
                         if now - req_time < self.time_window_seconds]

        return len(self.requests) < self.max_requests

    def record_request(self):
        """Record a new request"""
        self.requests.append(time.time())

    def time_until_next_request(self) -> float:
        """Get seconds to wait before next request"""
        if self.can_make_request():
            return 0.0

        if not self.requests:
            return 0.0

        oldest_request = min(self.requests)
        return self.time_window_seconds - (time.time() - oldest_request)


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""

    failure_threshold: int = 5
    reset_timeout_seconds: int = 60
    failures: int = 0
    last_failure: Optional[float] = None
    state: str = "closed"  # closed, open, half_open

    def can_make_request(self) -> bool:
        """Check if circuit breaker allows requests"""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure and time.time() - self.last_failure > self.reset_timeout_seconds:
                self.state = "half_open"
                return True
            return False

        # half_open state - allow one request to test
        return True

    def record_success(self):
        """Record successful request"""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed request"""
        self.failures += 1
        self.last_failure = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"


class BaseAPIClient(ABC):
    """Base class for all API clients with common functionality"""

    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 rate_limit_per_minute: int = 60,
                 timeout_seconds: int = 30,
                 enable_circuit_breaker: bool = True):

        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout_seconds
        self.settings = get_settings()

        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit_per_minute,
            time_window_seconds=60
        )

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        # Client state
        self.status = ClientStatus.HEALTHY
        self.last_request_time: Optional[datetime] = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for requests"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"RestaurantRecommender/1.0.0"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def _make_request(self,
                            method: str,
                            endpoint: str,
                            params: Optional[Dict[str, Any]] = None,
                            data: Optional[Dict[str, Any]] = None,
                            custom_headers: Optional[Dict[str, str]] = None) -> APIResponse[Dict[str, Any]]:
        """Make HTTP request with error handling and rate limiting"""

        start_time = time.time()

        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_make_request():
            self.status = ClientStatus.UNAVAILABLE
            return APIResponse.error_response("Service temporarily unavailable (circuit breaker open)")

        # Check rate limiting
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.time_until_next_request()
            self.status = ClientStatus.RATE_LIMITED
            return APIResponse.error_response(
                f"Rate limit exceeded. Retry in {wait_time:.1f} seconds",
                status_code=429
            )

        # Record request attempt
        self.rate_limiter.record_request()
        self.total_requests += 1
        self.last_request_time = datetime.utcnow()

        try:
            # Prepare request
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = self._get_headers()
            if custom_headers:
                headers.update(custom_headers)

            session = await self._get_session()

            # Make request
            async with session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=data,
                    headers=headers
            ) as response:

                response_time_ms = (time.time() - start_time) * 1000

                # Handle response
                if response.status == 200:
                    response_data = await response.json()
                    self.successful_requests += 1
                    self.status = ClientStatus.HEALTHY

                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()

                    return APIResponse.success_response(
                        data=response_data,
                        response_time_ms=response_time_ms
                    )

                elif response.status == 429:
                    self.status = ClientStatus.RATE_LIMITED
                    error_msg = "Rate limit exceeded"

                    # Try to get retry-after header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        error_msg += f". Retry after {retry_after} seconds"

                    return APIResponse.error_response(error_msg, response.status)

                else:
                    error_text = await response.text()
                    self.failed_requests += 1
                    self.status = ClientStatus.DEGRADED

                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()

                    return APIResponse.error_response(
                        f"API request failed: {error_text}",
                        response.status
                    )

        except asyncio.TimeoutError:
            self.failed_requests += 1
            self.status = ClientStatus.DEGRADED

            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            return APIResponse.error_response("Request timeout")

        except Exception as e:
            self.failed_requests += 1
            self.status = ClientStatus.DEGRADED

            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            logger.error(f"API request failed: {str(e)}")
            return APIResponse.error_response(f"Request failed: {str(e)}")

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse[Dict[str, Any]]:
        """Make GET request"""
        return await self._make_request("GET", endpoint, params=params)

    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse[Dict[str, Any]]:
        """Make POST request"""
        return await self._make_request("POST", endpoint, data=data)

    def get_health_status(self) -> Dict[str, Any]:
        """Get client health information"""
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

        return {
            "status": self.status.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "rate_limit_status": {
                "can_make_request": self.rate_limiter.can_make_request(),
                "requests_in_window": len(self.rate_limiter.requests),
                "max_requests_per_minute": self.rate_limiter.max_requests
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "failures": self.circuit_breaker.failures if self.circuit_breaker else 0
            } if self.circuit_breaker else None
        }

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the API is healthy"""
        pass


class CacheableAPIClient(BaseAPIClient):
    """Base class for API clients that support caching"""

    def __init__(self, cache_adapter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = cache_adapter

    async def _cached_request(self,
                              cache_key: str,
                              cache_ttl: int,
                              request_func: Callable[[], Any]) -> APIResponse[Any]:
        """Make request with caching support"""

        # Try cache first
        if self.cache:
            try:
                cached_data = await self.cache.get(cache_key)
                if cached_data is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return APIResponse.success_response(
                        data=cached_data,
                        cached=True
                    )
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Make actual request
        response = await request_func()

        # Cache successful responses
        if response.success and self.cache and response.data:
            try:
                await self.cache.set(cache_key, response.data, ttl=cache_ttl)
                logger.debug(f"Cached response for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

        return response


# Retry decorator for additional resilience
def with_retry(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator to add retry logic to methods"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        break

                    # Exponential backoff
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator