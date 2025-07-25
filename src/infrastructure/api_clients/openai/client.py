import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

from openai import AsyncOpenAI
from ..base_client import BaseAPIClient, APIResponse, with_retry
from ....config.settings import get_settings
from ....config.constants import OPENAI_EMBEDDING_DIMENSIONS, OPENAI_MAX_TOKENS
from ....models.common import CacheKey

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Chat message for OpenAI API"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatCompletion:
    """Chat completion response"""
    content: str
    tokens_used: int
    model: str
    finish_reason: str
    response_time_ms: float


@dataclass
class EmbeddingResult:
    """Embedding response"""
    embedding: List[float]
    dimensions: int
    model: str
    tokens_used: int
    response_time_ms: float


class OpenAIClient(BaseAPIClient):
    """OpenAI API client with caching and error handling"""

    def __init__(self, api_key: Optional[str] = None, cache_adapter=None):
        settings = get_settings()

        # Use provided key or get from settings
        self.api_key = api_key or settings.api.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=self.api_key,
            rate_limit_per_minute=settings.api.api_rate_limits.get("openai", 60),
            timeout_seconds=60
        )

        # OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.cache = cache_adapter
        self.settings = settings

        # Token usage tracking
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0

    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            # Simple test with minimal tokens
            response = await self.chat_completion(
                messages=[ChatMessage(role="user", content="Hi")],
                model="gpt-3.5-turbo",
                max_tokens=1
            )
            return response.success
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

    @with_retry(max_retries=2, backoff_factor=1.0)
    async def chat_completion(self,
                              messages: List[ChatMessage],
                              model: str = "gpt-3.5-turbo",
                              temperature: float = 0.1,
                              max_tokens: Optional[int] = None,
                              response_format: Optional[Dict[str, str]] = None,
                              use_cache: bool = True) -> APIResponse[ChatCompletion]:
        """Generate chat completion"""

        import time
        start_time = time.time()

        try:
            # Prepare cache key if caching enabled
            cache_key = None
            if use_cache and self.cache:
                cache_data = {
                    "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                cache_key = CacheKey(
                    prefix="openai_chat",
                    identifier=self._hash_dict(cache_data)
                )

                # Check cache
                cached_result = await self.cache.get(str(cache_key))
                if cached_result:
                    logger.debug(f"Cache hit for chat completion: {cache_key}")
                    return APIResponse.success_response(
                        data=ChatCompletion(**cached_result),
                        cached=True
                    )

            # Prepare messages for OpenAI
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Set default max_tokens if not provided
            if max_tokens is None:
                max_tokens = OPENAI_MAX_TOKENS.get(model, 1000)

            # Make API call
            kwargs = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)

            response_time_ms = (time.time() - start_time) * 1000

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            tokens_used = response.usage.total_tokens

            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost_estimate += self._estimate_cost(model, tokens_used)

            result = ChatCompletion(
                content=content,
                tokens_used=tokens_used,
                model=model,
                finish_reason=finish_reason,
                response_time_ms=response_time_ms
            )

            # Cache result if enabled
            if cache_key and self.cache:
                await self.cache.set(
                    str(cache_key),
                    result.__dict__,
                    ttl=self.settings.cache.ttl.get("openai_chat", 3600)
                )

            self.successful_requests += 1

            return APIResponse.success_response(
                data=result,
                response_time_ms=response_time_ms
            )

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"OpenAI chat completion failed: {e}")
            return APIResponse.error_response(f"Chat completion failed: {str(e)}")

    @with_retry(max_retries=2, backoff_factor=1.0)
    async def create_embedding(self,
                               text: str,
                               model: str = "text-embedding-3-large",
                               dimensions: Optional[int] = None,
                               use_cache: bool = True) -> APIResponse[EmbeddingResult]:
        """Create text embedding"""

        import time
        start_time = time.time()

        try:
            # Prepare cache key if caching enabled
            cache_key = None
            if use_cache and self.cache:
                cache_data = {
                    "text": text,
                    "model": model,
                    "dimensions": dimensions
                }
                cache_key = CacheKey(
                    prefix="openai_embedding",
                    identifier=self._hash_dict(cache_data)
                )

                # Check cache
                cached_result = await self.cache.get(str(cache_key))
                if cached_result:
                    logger.debug(f"Cache hit for embedding: {cache_key}")
                    return APIResponse.success_response(
                        data=EmbeddingResult(**cached_result),
                        cached=True
                    )

            # Set default dimensions
            if dimensions is None:
                dimensions = OPENAI_EMBEDDING_DIMENSIONS.get(model, 1536)

            # Make API call
            kwargs = {
                "model": model,
                "input": text
            }

            # Add dimensions for supported models
            if model in ["text-embedding-3-large", "text-embedding-3-small"]:
                kwargs["dimensions"] = dimensions

            response = await self.client.embeddings.create(**kwargs)

            response_time_ms = (time.time() - start_time) * 1000

            # Extract embedding data
            embedding_data = response.data[0]
            embedding = embedding_data.embedding
            tokens_used = response.usage.total_tokens

            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost_estimate += self._estimate_embedding_cost(model, tokens_used)

            result = EmbeddingResult(
                embedding=embedding,
                dimensions=len(embedding),
                model=model,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms
            )

            # Cache result if enabled
            if cache_key and self.cache:
                await self.cache.set(
                    str(cache_key),
                    result.__dict__,
                    ttl=self.settings.cache.ttl.get("embeddings", 86400)  # 24 hours
                )

            self.successful_requests += 1

            return APIResponse.success_response(
                data=result,
                response_time_ms=response_time_ms
            )

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"OpenAI embedding failed: {e}")
            return APIResponse.error_response(f"Embedding creation failed: {str(e)}")

    async def create_embeddings_batch(self,
                                      texts: List[str],
                                      model: str = "text-embedding-3-large",
                                      dimensions: Optional[int] = None,
                                      batch_size: int = 100) -> APIResponse[List[EmbeddingResult]]:
        """Create embeddings for multiple texts in batches"""

        all_results = []
        total_response_time = 0.0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Join batch texts for single API call
                batch_text = "\n".join(batch)
                response = await self.create_embedding(
                    text=batch_text,
                    model=model,
                    dimensions=dimensions,
                    use_cache=False  # Don't cache batch requests
                )

                if not response.success:
                    return response

                total_response_time += response.response_time_ms

                # For simplicity, create individual results
                # In production, you'd want proper batch handling
                for j, text in enumerate(batch):
                    individual_response = await self.create_embedding(
                        text=text,
                        model=model,
                        dimensions=dimensions
                    )
                    if individual_response.success:
                        all_results.append(individual_response.data)

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                return APIResponse.error_response(f"Batch embedding failed: {str(e)}")

        return APIResponse.success_response(
            data=all_results,
            response_time_ms=total_response_time
        )

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create hash from dictionary for caching"""
        import hashlib
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _estimate_cost(self, model: str, tokens: int) -> float:
        """Estimate cost for chat completion"""
        # Simplified cost estimation (as of 2024)
        cost_per_1k_tokens = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01
        }

        rate = cost_per_1k_tokens.get(model, 0.0015)
        return (tokens / 1000) * rate

    def _estimate_embedding_cost(self, model: str, tokens: int) -> float:
        """Estimate cost for embeddings"""
        cost_per_1k_tokens = {
            "text-embedding-3-large": 0.00013,
            "text-embedding-3-small": 0.00002
        }

        rate = cost_per_1k_tokens.get(model, 0.00013)
        return (tokens / 1000) * rate

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost_usd": round(self.total_cost_estimate, 4),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / max(self.total_requests, 1)) * 100
        }