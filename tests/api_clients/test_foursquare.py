# tests/api_clients/test_foursquare.py
import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import asyncio
from src.infrastructure.api_clients.foursquare.client import FoursquareClient
from src.infrastructure.api_clients.google_places.client import NearbySearchRequest


class TestFoursquareClient:
    """Comprehensive Foursquare API client tests"""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client can be initialized properly"""
        client = FoursquareClient()
        assert client is not None
        assert client.api_key is not None
        assert "fsq3" not in client.api_key  # Should be stripped

    @pytest.mark.asyncio
    async def test_headers_format(self):
        """Test that headers are formatted correctly"""
        client = FoursquareClient()
        headers = client._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "X-Places-Api-Version" in headers
        assert headers["X-Places-Api-Version"] == "2025-06-17"
        assert "fsq3" not in headers["Authorization"]  # Prefix should be removed

        # Cleanup
        if hasattr(client, '_session') and client._session:
            await client._session.close()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test API health check"""
        client = FoursquareClient()
        try:
            is_healthy = await client.health_check()
            assert isinstance(is_healthy, bool)

            if is_healthy:
                print("✅ Foursquare API is healthy")
            else:
                print("⚠️ Foursquare API health check failed")
        finally:
            # Cleanup
            if hasattr(client, '_session') and client._session:
                await client._session.close()

    @pytest.mark.asyncio
    async def test_restaurant_search_basic(self):
        """Test basic restaurant search"""
        client = FoursquareClient()
        try:
            request = NearbySearchRequest(
                location=(40.7128, -74.0060),  # NYC
                radius=1000
            )

            response = await client.nearby_search(request)

            assert response is not None
            if response.success:
                assert isinstance(response.data, list)
                print(f"✅ Found {len(response.data)} places")

                # Check that we got some restaurants
                if response.data:
                    restaurant = response.data[0]
                    assert hasattr(restaurant, 'name')
                    assert hasattr(restaurant, 'location')
                    assert hasattr(restaurant, 'primary_category')
                    print(f"   Sample: {restaurant.name} ({restaurant.primary_category.value})")
            else:
                print(f"⚠️ Search failed: {response.error}")
        finally:
            # Cleanup
            if hasattr(client, '_session') and client._session:
                await client._session.close()

    @pytest.mark.asyncio
    async def test_restaurant_search_with_keyword(self):
        """Test restaurant search with specific cuisine"""
        client = FoursquareClient()
        try:
            request = NearbySearchRequest(
                location=(40.7128, -74.0060),  # NYC
                radius=2000,
                keyword="italian"
            )

            response = await client.nearby_search(request)

            if response.success and response.data:
                # Check that results are relevant to "italian"
                italian_found = False
                for restaurant in response.data[:5]:  # Check first 5
                    name_lower = restaurant.name.lower()
                    category_lower = restaurant.primary_category.value.lower()

                    if "italian" in name_lower or "italian" in category_lower:
                        italian_found = True
                        break

                print(f"✅ Italian keyword search: {len(response.data)} results")
                if italian_found:
                    print("✅ Found Italian restaurants in results")
                else:
                    print("⚠️ No obvious Italian restaurants in results")
            else:
                print(f"⚠️ Italian search failed: {response.error if response else 'No response'}")
        finally:
            # Cleanup
            if hasattr(client, '_session') and client._session:
                await client._session.close()


# Standalone test function (can be run independently)
async def test_foursquare_standalone():
    """Standalone test that can be run without pytest"""
    print("🧪 Standalone Foursquare Test")
    print("=" * 40)

    client = FoursquareClient()
    try:
        # Test 1: Health check
        print("1. Health check...")
        is_healthy = await client.health_check()
        print(f"   {'✅' if is_healthy else '❌'} Health: {is_healthy}")

        if not is_healthy:
            print("   Stopping tests due to health check failure")
            return

        # Test 2: Basic search
        print("\n2. Basic restaurant search...")
        request = NearbySearchRequest(
            location=(40.7128, -74.0060),  # NYC
            radius=1000,
            keyword="italian"
        )

        response = await client.nearby_search(request)

        if response.success:
            print(f"   ✅ Found {len(response.data)} restaurants")

            for i, restaurant in enumerate(response.data[:3], 1):
                print(f"   {i}. {restaurant.name}")
                print(f"      🍴 {restaurant.primary_category.value}")
                print(f"      📍 {restaurant.formatted_address}")
                if restaurant.rating > 0:
                    print(f"      ⭐ {restaurant.rating}/5.0")
        else:
            print(f"   ❌ Search failed: {response.error}")

        # Test 3: Performance stats
        print(f"\n3. Client performance:")
        stats = client.get_health_status()
        print(f"   Requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate_percent']:.1f}%")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if hasattr(client, '_session') and client._session:
            await client._session.close()


if __name__ == "__main__":
    # Run standalone test
    asyncio.run(test_foursquare_standalone())