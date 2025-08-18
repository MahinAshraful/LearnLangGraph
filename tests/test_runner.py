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

        # Test Google Places (real API if key available, otherwise mock)
        await self.test_google_places_api()

        # Test OpenAI (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            await self.test_openai_parsing()
        else:
            print("⚠️  OpenAI API key not found - skipping LLM tests")
            self.results["skipped"] += 1

    async def test_google_places_api(self):
        """Test Google Places API integration"""
        try:
            from src.infrastructure.api_clients.google_places.client import GooglePlacesClient, NearbySearchRequest

            # Determine if we should use real API or mock
            api_key = os.getenv("GOOGLE_PLACES_API_KEY")
            if api_key:
                print("🌐 Testing Google Places API with real API key")
                client = GooglePlacesClient(api_key=api_key, use_mock=False)
                test_mode = "Real API"
            else:
                print("🎭 Testing Google Places API with mock data")
                client = GooglePlacesClient(use_mock=True)
                test_mode = "Mock"

            # Health check (for real API only)
            if not client.use_mock:
                is_healthy = await client.health_check()
                assert is_healthy, "Google Places API health check failed"

            # Search test
            request = NearbySearchRequest(
                location=(40.7128, -74.0060),
                radius=1000,
                keyword="italian"
            )

            response = await client.nearby_search(request)
            assert response.success, f"Google Places search failed: {response.error}"
            assert len(response.data) > 0, "No restaurants found"

            # Cleanup
            if hasattr(client, '_session') and client._session:
                await client._session.close()

            print(f"✅ Google Places API tests passed ({test_mode})")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Google Places API test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_openai_parsing(self):
        """Test OpenAI query parsing"""
        try:
            from src.infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage

            client = OpenAIClient()

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
        """Test cache system functionality"""
        try:
            from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter

            cache = MemoryAdapter()
            await cache.connect()

            # Test basic cache operations
            await cache.set("test_key", "test_value", ttl=60)
            value = await cache.get("test_key")
            assert value == "test_value", "Cache get/set failed"

            # Test cache expiry
            await cache.set("expire_key", "expire_value", ttl=1)
            await asyncio.sleep(1.1)
            expired_value = await cache.get("expire_key")
            assert expired_value is None, "Cache expiry failed"

            await cache.disconnect()

            print("✅ Cache system tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Cache system test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_agents(self):
        """Test core agent functionality"""
        await self.test_workflow_creation()
        await self.test_end_to_end_recommendation()

    async def test_workflow_creation(self):
        """Test workflow can be created"""
        try:
            from src.agents.workflows.restaurant_recommendation import create_restaurant_recommendation_workflow
            from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter

            cache = MemoryAdapter()
            await cache.connect()

            # Create workflow with mock services
            workflow = await create_restaurant_recommendation_workflow(
                openai_api_key="fake_key_for_testing",
                use_mock_services=True,
                cache_adapter=cache
            )

            assert workflow is not None, "Workflow creation failed"
            assert hasattr(workflow, 'query_parser'), "Query parser missing"
            assert hasattr(workflow, 'scoring'), "Scoring agent missing"

            await cache.disconnect()

            print("✅ Workflow creation tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Workflow creation test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_end_to_end_recommendation(self):
        """Test complete recommendation flow"""
        try:
            from src.agents.workflows.restaurant_recommendation import create_restaurant_recommendation_workflow
            from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter

            cache = MemoryAdapter()
            await cache.connect()

            # Create workflow
            workflow = await create_restaurant_recommendation_workflow(
                openai_api_key="fake_key_for_testing",
                use_mock_services=True,
                cache_adapter=cache
            )

            # Test recommendation
            result = await workflow.recommend_restaurants(
                user_query="Find me Italian dinner for 2",
                user_id="test_user",
                user_location=(40.7128, -74.0060)
            )

            assert result["success"], f"Recommendation failed: {result.get('error', 'Unknown error')}"
            assert len(result["data"]["recommendations"]) > 0, "No recommendations returned"

            await cache.disconnect()

            print("✅ End-to-end recommendation tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ End-to-end recommendation test failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_models(self):
        """Test data models"""
        try:
            from src.models.restaurant import Restaurant, RestaurantCategory, PriceLevel
            from src.models.recommendation import Recommendation, ScoreBreakdown
            from src.models.query import ParsedQuery, QueryType

            # Test basic model creation
            restaurant = Restaurant(
                place_id="test_123",
                name="Test Restaurant",
                primary_category=RestaurantCategory.ITALIAN,
                rating=4.5,
                price_level=PriceLevel.MODERATE
            )

            assert restaurant.name == "Test Restaurant"
            assert restaurant.rating == 4.5

            print("✅ Data model tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Data model tests failed: {e}")
            self.results["failed"] += 1
            raise

    async def test_integration(self):
        """Test integration scenarios"""
        try:
            # Test places factory
            from src.infrastructure.api_clients.places_factory import PlacesClientFactory

            # Test auto selection
            client = PlacesClientFactory.create_client("auto")
            assert client is not None, "Factory failed to create client"

            # Test mock selection
            mock_client = PlacesClientFactory.create_client("mock")
            assert mock_client is not None, "Factory failed to create mock client"
            assert mock_client.use_mock == True, "Mock client not properly configured"

            print("✅ Integration tests passed")
            self.results["passed"] += 1

        except Exception as e:
            print(f"❌ Integration tests failed: {e}")
            self.results["failed"] += 1
            raise

    def print_summary(self, total_time: float):
        """Print test summary"""
        total_tests = self.results["passed"] + self.results["failed"] + self.results["skipped"]
        success_rate = (self.results["passed"] / total_tests * 100) if total_tests > 0 else 0

        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"⚠️  Skipped: {self.results['skipped']}")
        print(f"📈 Total: {total_tests}")
        print(f"⏱️  Time: {total_time:.2f}s")
        print(f"🎯 Success Rate: {success_rate:.1f}%")

        if self.results["errors"]:
            print(f"\n❌ ERRORS:")
            for error in self.results["errors"]:
                print(f"  - {error}")

        if self.results["failed"] == 0:
            print(f"\n🎉 All tests passed! System is ready for Google Places.")
        else:
            print(f"\n⚠️  Some tests failed. Please review and fix issues.")


async def main():
    """Run the test suite"""
    runner = TestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())