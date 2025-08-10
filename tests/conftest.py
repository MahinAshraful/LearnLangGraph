import pytest
import asyncio
import os
import sys
from typing import AsyncGenerator

# Add src to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter
from src.infrastructure.databases.vector_db.mock_adapter import MockVectorAdapter
from src.infrastructure.api_clients.foursquare.client import FoursquareClient
from src.infrastructure.api_clients.google_places.client import GooglePlacesClient
from src.infrastructure.api_clients.openai.client import OpenAIClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def cache_adapter() -> AsyncGenerator[MemoryAdapter, None]:
    """Provide a clean cache adapter for tests"""
    cache = MemoryAdapter(key_prefix="test")
    await cache.connect()
    yield cache
    await cache.disconnect()


@pytest.fixture
async def vector_db() -> AsyncGenerator[MockVectorAdapter, None]:
    """Provide a clean vector database for tests"""
    vector_db = MockVectorAdapter("test_recommendations")
    await vector_db.connect()
    yield vector_db
    await vector_db.disconnect()


@pytest.fixture
async def foursquare_client(cache_adapter) -> AsyncGenerator[FoursquareClient, None]:
    """Provide Foursquare client (only if API key available)"""
    if not os.getenv("FOURSQUARE_API_KEY"):
        pytest.skip("FOURSQUARE_API_KEY not available")

    client = FoursquareClient(cache_adapter=cache_adapter)
    yield client

    # Cleanup
    if hasattr(client, '_session') and client._session:
        await client._session.close()


@pytest.fixture
async def mock_places_client(cache_adapter) -> AsyncGenerator[GooglePlacesClient, None]:
    """Provide mock Google Places client"""
    client = GooglePlacesClient(cache_adapter=cache_adapter, use_mock=True)
    yield client

    # Cleanup
    if hasattr(client, '_session') and client._session:
        await client._session.close()


@pytest.fixture
async def openai_client(cache_adapter) -> AsyncGenerator[OpenAIClient, None]:
    """Provide OpenAI client (only if API key available)"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available")

    client = OpenAIClient(cache_adapter=cache_adapter)
    yield client

    # Cleanup
    if hasattr(client, '_session') and client._session:
        await client._session.close()


# Test data fixtures
@pytest.fixture
def sample_queries():
    """Sample restaurant queries for testing"""
    return [
        "Find me a good Italian restaurant for dinner tonight",
        "I want cheap tacos for 4 people",
        "Looking for expensive Japanese restaurant for business dinner",
        "Vegetarian restaurant with outdoor seating near me",
        "Quick lunch spot with parking downtown"
    ]


@pytest.fixture
def sample_location():
    """Sample NYC location for testing"""
    return (40.7128, -74.0060)  # NYC coordinates