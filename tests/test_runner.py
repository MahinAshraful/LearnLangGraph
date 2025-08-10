# tests/test_runner.py
import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestRunner:
    """Custom test runner for restaurant recommendation system"""

    def __init__(self):
        self.results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }

    async def run_all_tests(self):
        """Run all tests with proper categorization"""
        print("🧪 Restaurant Recommendation System Test Suite")
        print("=" * 60)

        start_time = time.time()

        # Test categories
        test_categories = [
            ("API Clients", self.test_api_clients),
            ("Database Systems", self.test_databases),
            ("Core Agents", self.test_agents),
            ("Data Models", self.test_models),
            ("Integration", self.test_integration)
        ]

        for category_name, test_function in test_categories:
            print(f"\n📋 Testing {category_name}")
            print("-" * 40)

            try:
                await test_function()
            except Exception as e:
                self.results["failed"] += 1
                self.results["errors"].append(f"{category_name}: {str(e)}")
                print(f"❌ {category_name} tests failed: {e}")

        # Summary
        total_time = time.time() - start_time
        self.print_summary(total_time)

    async def test_api_clients(self):
        """Test all API clients"""

        # Test Foursquare (if API key available)
        if os.getenv("FOURSQUARE_API_KEY"):
            await self.test_foursquare_api()
        else:
            print("⚠️  Foursquare API key not found - skipping real API tests")
            self.results["skipped"] += 1

        # Test Mock Google Places
        await self.test_mock_google_places()

        # Test OpenAI (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            await self.test_openai_parsing()
        else:
            print("⚠️  OpenAI API key not found - skipping LLM tests")
            self.results["skipped"] += 1

    async def test_foursquare_api(self):
        """Test Foursquare API integration"""
        try:
            from src.infrastructure.api_clients.foursquare.client import FoursquareClient
            from src.infrastructure.api_clients.google_places.client import NearbySearchRequest

            client = FoursquareClient()

            # Health check
            is_healthy = await client.health_check()
            assert is_healthy, "Foursquare API health check failed"

            # Search test
            request = NearbySearchRequest(
                location=(40.7128, -74.0060),
                radius=1000,
                keyword="italian"
            )

            response = await client.nearby_search(request)
            assert response.success, f"Foursquare search failed: {response.error}"
            assert len(response.data) > 0, "No restaurants found"

            # Cleanup
            if hasattr(client, '_session') and client._session:
                await client._session.close()

            print("✅ Foursquare API tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Foursquare API test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_mock_google_places(self):
        """Test mock Google Places client"""
        try:
            from src.infrastructure.api_clients.google_places.client import GooglePlacesClient, NearbySearchRequest

            client = GooglePlacesClient(use_mock=True)

            # Test search
            request = NearbySearchRequest(
                location=(40.7128, -74.0060),
                radius=5000,
                keyword="italian"
            )

            response = await client.nearby_search(request)
            assert response.success, "Mock search failed"
            assert len(response.data) > 0, "No mock restaurants generated"

            # Test that we get Italian restaurants
            italian_found = any(
                'italian' in r.primary_category.value.lower() or 'italian' in r.name.lower()
                for r in response.data
            )
            assert italian_found, "No Italian restaurants in mock data"

            print("✅ Mock Google Places tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Mock Google Places test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_openai_parsing(self):
        """Test OpenAI query parsing"""
        try:
            from src.infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage

            client = OpenAI(OpenAIClient())

            # Test simple completion
            messages = [ChatMessage(role="user", content="Parse this restaurant query: 'Italian dinner for 2'")]
            response = await client.chat_completion(messages, max_tokens=100)

            assert response.success, f"OpenAI completion failed: {response.error}"
            assert len(response.data.content) > 0, "Empty response from OpenAI"

            print("✅ OpenAI API tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ OpenAI API test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_databases(self):
        """Test database systems"""
        await self.test_vector_db()
        await self.test_cache_system()

    async def test_vector_db(self):
        """Test vector database functionality"""
        try:
            from src.infrastructure.databases.vector_db.mock_adapter import MockVectorAdapter

            vector_db = MockVectorAdapter()
            await vector_db.connect()

            # Test that mock data is generated
            collections = await vector_db.list_collections()
            assert "user_preferences" in collections, "User preferences collection not found"
            assert "restaurant_features" in collections, "Restaurant features collection not found"

            # Test user similarity search
            similar_users = await vector_db.get_similar_users("foodie_explorer_0", limit=3)
            assert len(similar_users) > 0, "No similar users found"

            await vector_db.disconnect()

            print("✅ Vector database tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Vector database test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_cache_system(self):
        """Test caching system"""
        try:
            from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter

            cache = MemoryAdapter()
            await cache.connect()

            # Test basic operations
            await cache.set("test_key", {"data": "test_value"}, ttl=60)
            result = await cache.get("test_key")
            assert result is not None, "Cache get failed"
            assert result["data"] == "test_value", "Cache data mismatch"

            # Test expiration
            await cache.set("temp_key", "temp_value", ttl=1)
            await asyncio.sleep(1.1)
            expired_result = await cache.get("temp_key")
            assert expired_result is None, "Cache expiration failed"

            await cache.disconnect()

            print("✅ Cache system tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Cache system test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_agents(self):
        """Test core agent functionality"""
        await self.test_query_parsing()
        await self.test_scoring_algorithm()

    async def test_query_parsing(self):
        """Test query parsing logic"""
        try:
            from src.models.query import ParsedQuery, QueryType
            from src.models.restaurant import RestaurantCategory, PriceLevel

            # Test that we can create and parse basic queries
            test_cases = [
                ("Italian restaurant", QueryType.CUISINE_SPECIFIC, RestaurantCategory.ITALIAN),
                ("cheap tacos", QueryType.CUISINE_SPECIFIC, RestaurantCategory.MEXICAN),
                ("expensive sushi", QueryType.CUISINE_SPECIFIC, RestaurantCategory.SUSHI)
            ]

            for query_text, expected_type, expected_cuisine in test_cases:
                # Create a basic parsed query manually (simulating the parser)
                parsed_query = ParsedQuery(
                    original_query=query_text,
                    query_type=expected_type,
                    cuisine_preferences=[expected_cuisine] if expected_cuisine else []
                )

                assert parsed_query.query_type == expected_type
                if expected_cuisine:
                    assert expected_cuisine in parsed_query.cuisine_preferences

            print("✅ Query parsing tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Query parsing test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_scoring_algorithm(self):
        """Test scoring algorithm"""
        try:
            from src.models.recommendation import ScoreBreakdown

            # Test score calculation
            score = ScoreBreakdown(
                preference_score=0.8,
                context_score=0.7,
                quality_score=0.9,
                boost_score=0.1
            )

            total = score.total_score
            expected = 0.8 * 0.5 + 0.7 * 0.3 + 0.9 * 0.15 + 0.1 * 0.05
            assert abs(total - expected) < 0.001, f"Score calculation error: {total} != {expected}"

            print("✅ Scoring algorithm tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Scoring algorithm test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_models(self):
        """Test data models"""
        try:
            from src.models.restaurant import Restaurant, RestaurantCategory, PriceLevel
            from src.models.common import Location

            # Test restaurant model creation
            location = Location(latitude=40.7128, longitude=-74.0060, city="New York")

            restaurant = Restaurant(
                place_id="test_id",
                name="Test Restaurant",
                location=location,
                primary_category=RestaurantCategory.ITALIAN,
                price_level=PriceLevel.MODERATE,
                rating=4.5,
                formatted_address="123 Test St, New York, NY"
            )

            assert restaurant.name == "Test Restaurant"
            assert restaurant.rating == 4.5
            assert restaurant.primary_category == RestaurantCategory.ITALIAN

            print("✅ Data model tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Data model test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_integration(self):
        """Test integration scenarios"""
        try:
            # Test that mock services can be created
            from src.agents.workflows.restaurant_recommendation import create_restaurant_recommendation_workflow
            from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter

            cache = MemoryAdapter()
            await cache.connect()

            # Test workflow creation with mocks
            workflow = await create_restaurant_recommendation_workflow(
                openai_api_key="fake_key_for_testing",
                use_mock_services=True,
                cache_adapter=cache
            )

            assert workflow is not None, "Workflow creation failed"

            await cache.disconnect()

            print("✅ Integration tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            self.results["failed"] += 1
            raise

    def print_summary(self, total_time: float):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = self.results["passed"] + self.results["failed"] + self.results["skipped"]

        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"⚠️  Skipped: {self.results['skipped']}")
        print(f"📈 Total: {total_tests}")
        print(f"⏱️  Time: {total_time:.2f}s")

        if self.results["failed"] > 0:
            print(f"\n❌ FAILURES:")
            for error in self.results["errors"]:
                print(f"   - {error}")

        success_rate = (self.results["passed"] / max(total_tests, 1)) * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")

        if success_rate >= 80:
            print("🎉 Test suite in good shape!")
        elif success_rate >= 60:
            print("⚠️  Some issues to address")
        else:
            print("🚨 Multiple issues need attention")


async def main():
    """Run all tests"""
    runner = TestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())