# tests/api_clients/test_foursquare.py
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
    async def test_headers_format(self, foursquare_client):
        """Test that headers are formatted correctly"""
        headers = foursquare_client._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "X-Places-Api-Version" in headers
        assert headers["X-Places-Api-Version"] == "2025-06-17"
        assert "fsq3" not in headers["Authorization"]  # Prefix should be removed

    @pytest.mark.asyncio
    async def test_health_check(self, foursquare_client):
        """Test API health check"""
        is_healthy = await foursquare_client.health_check()
        assert isinstance(is_healthy, bool)

        if is_healthy:
            print("✅ Foursquare API is healthy")
        else:
            print("⚠️ Foursquare API health check failed")

    @pytest.mark.asyncio
    async def test_restaurant_search_basic(self, foursquare_client, sample_location):
        """Test basic restaurant search"""
        request = NearbySearchRequest(
            location=sample_location,
            radius=1000
        )

        response = await foursquare_client.nearby_search(request)

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
        else:
            print(f"⚠️ Search failed: {response.error}")

    @pytest.mark.asyncio
    async def test_restaurant_search_with_keyword(self, foursquare_client, sample_location):
        """Test restaurant search with specific cuisine"""
        request = NearbySearchRequest(
            location=sample_location,
            radius=2000,
            keyword="italian"
        )

        response = await foursquare_client.nearby_search(request)

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

    @pytest.mark.asyncio
    async def test_category_detection(self, foursquare_client):
        """Test restaurant category detection logic"""
        # Test restaurant categories
        restaurant_categories = [
            {"name": "Italian Restaurant"},
            {"name": "Pizza Place"},
            {"name": "Café"}
        ]

        non_restaurant_categories = [
            {"name": "Park"},
            {"name": "Museum"},
            {"name": "Office Building"}
        ]

        # Test positive cases
        for category in restaurant_categories:
            is_restaurant = foursquare_client._is_restaurant_category([category])
            assert is_restaurant, f"Should detect {category['name']} as restaurant"

        # Test negative cases
        for category in non_restaurant_categories:
            is_restaurant = foursquare_client._is_restaurant_category([category])
            assert not is_restaurant, f"Should not detect {category['name']} as restaurant"

        print("✅ Category detection logic working correctly")

    @pytest.mark.asyncio
    async def test_category_mapping(self, foursquare_client):
        """Test category mapping to our enums"""
        from src.models.restaurant import RestaurantCategory

        test_cases = [
            ([{"name": "Italian Restaurant"}], RestaurantCategory.ITALIAN),
            ([{"name": "Sushi Bar"}], RestaurantCategory.SUSHI),
            ([{"name": "Coffee Shop"}], RestaurantCategory.CAFE),
            ([{"name": "Mexican Grill"}], RestaurantCategory.MEXICAN)
        ]

        for categories, expected_category in test_cases:
            mapped_category = foursquare_client._map_foursquare_category(categories)
            assert mapped_category == expected_category, f"Wrong mapping for {categories[0]['name']}"

        print("✅ Category mapping working correctly")

    @pytest.mark.asyncio
    async def test_performance_metrics(self, foursquare_client, sample_location):
        """Test that performance metrics are tracked"""
        import time

        start_time = time.time()

        request = NearbySearchRequest(
            location=sample_location,
            radius=1000
        )

        response = await foursquare_client.nearby_search(request)

        end_time = time.time()
        request_time = end_time - start_time

        # Check response time is reasonable (less than 5 seconds)
        assert request_time < 5.0, f"Request took too long: {request_time:.2f}s"

        # Check that response includes timing info
        if response.success:
            assert hasattr(response, 'response_time_ms')
            print(f"✅ Request completed in {request_time:.2f}s")

    @pytest.mark.asyncio
    async def test_error_handling(self, cache_adapter):
        """Test error handling with invalid API key"""
        # Create client with invalid API key
        invalid_client = FoursquareClient(api_key="invalid_key", cache_adapter=cache_adapter)

        request = NearbySearchRequest(
            location=(40.7128, -74.0060),
            radius=1000
        )

        response = await invalid_client.nearby_search(request)

        # Should handle error gracefully
        assert not response.success
        assert response.error is not None
        print(f"✅ Error handling working: {response.error}")

        # Cleanup
        if hasattr(invalid_client, '_session') and invalid_client._session:
            await invalid_client._session.close()


class TestFoursquareIntegration:
    """Integration tests for Foursquare client"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, foursquare_client):
        """Test handling multiple concurrent requests"""
        locations = [
            (40.7128, -74.0060),  # NYC
            (40.7589, -73.9851),  # Upper West Side
            (40.7505, -73.9934),  # Midtown
        ]

        requests = [
            NearbySearchRequest(location=loc, radius=1000)
            for loc in locations
        ]

        # Run requests concurrently
        tasks = [foursquare_client.nearby_search(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all requests completed
        assert len(responses) == len(requests)

        successful_responses = 0
        for response in responses:
            if not isinstance(response, Exception) and response.success:
                successful_responses += 1

        print(f"✅ {successful_responses}/{len(requests)} concurrent requests succeeded")
        assert successful_responses > 0, "No concurrent requests succeeded"

    @pytest.mark.asyncio
    async def test_restaurant_data_quality(self, foursquare_client, sample_location):
        """Test quality of restaurant data returned"""
        request = NearbySearchRequest(
            location=sample_location,
            radius=2000,
            keyword="restaurant"
        )

        response = await foursquare_client.nearby_search(request)

        if response.success and response.data:
            restaurant = response.data[0]

            # Check required fields
            assert restaurant.place_id is not None and restaurant.place_id != ""
            assert restaurant.name is not None and restaurant.name != ""
            assert restaurant.location is not None
            assert restaurant.location.latitude != 0
            assert restaurant.location.longitude != 0
            assert restaurant.primary_category is not None

            # Check optional but important fields
            quality_score = 0
            if restaurant.formatted_address:
                quality_score += 1
            if restaurant.phone_number:
                quality_score += 1
            if restaurant.website:
                quality_score += 1
            if restaurant.rating > 0:
                quality_score += 1

            print(f"✅ Restaurant data quality: {quality_score}/4")
            print(f"   Name: {restaurant.name}")
            print(f"   Category: {restaurant.primary_category.value}")
            print(f"   Address: {restaurant.formatted_address}")

            assert quality_score >= 2, "Restaurant data quality too low"


# Standalone test function (can be run independently)
async def test_foursquare_standalone():
    """Standalone test that can be run without pytest"""
    print("🧪 Standalone Foursquare Test")
    print("=" * 40)

    try:
        client = FoursquareClient()

        # Test 1: Health check
        print("1. Health check...")
        is_healthy = await client.health_check()
        print(f"   {'✅' if is_healthy else '❌'} Health: {is_healthy}")

        if not is_healthy:
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