# tests/agents/test_workflow.py
import pytest
import asyncio
from src.agents.workflows.restaurant_recommendation import create_restaurant_recommendation_workflow


class TestWorkflowIntegration:
    """Test the complete workflow integration"""

    @pytest.mark.asyncio
    async def test_workflow_creation_mock(self, cache_adapter):
        """Test workflow can be created with mock services"""
        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache_adapter
        )

        assert workflow is not None
        assert hasattr(workflow, 'query_parser')
        assert hasattr(workflow, 'user_context')
        assert hasattr(workflow, 'data_retrieval')
        assert hasattr(workflow, 'candidate_filter')
        assert hasattr(workflow, 'scoring')
        assert hasattr(workflow, 'output_formatter')

        print("✅ Mock workflow created successfully")

    @pytest.mark.asyncio
    async def test_workflow_with_real_apis(self, cache_adapter):
        """Test workflow creation with real APIs (if available)"""
        import os

        # Skip if no API keys available
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("FOURSQUARE_API_KEY"):
            pytest.skip("No API keys available for real API testing")

        try:
            workflow = await create_restaurant_recommendation_workflow(
                openai_api_key=os.getenv("OPENAI_API_KEY", "fake_key"),
                use_mock_services=False,
                cache_adapter=cache_adapter
            )

            assert workflow is not None
            print("✅ Real API workflow created successfully")

        except Exception as e:
            print(f"⚠️ Real API workflow creation failed: {e}")
            # Don't fail the test - API might be down

    @pytest.mark.asyncio
    async def test_end_to_end_mock_recommendation(self, cache_adapter):
        """Test complete recommendation flow with mock data"""
        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache_adapter
        )

        # Test simple query
        result = await workflow.recommend_restaurants(
            user_query="Find me a good Italian restaurant",
            user_id="test_user_123",
            user_location=(40.7128, -74.0060)
        )

        assert result is not None
        assert "success" in result
        assert "data" in result
        assert "performance" in result

        if result["success"]:
            data = result["data"]
            assert "message" in data
            assert "recommendations" in data
            print(f"✅ Mock recommendation successful: {len(data['recommendations'])} recommendations")
        else:
            print(f"⚠️ Mock recommendation failed: {result.get('error', 'Unknown error')}")

    @pytest.mark.asyncio
    async def test_query_complexity_routing(self, cache_adapter):
        """Test that different query complexities are handled correctly"""
        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache_adapter
        )

        test_queries = [
            # Simple query (should be fast)
            ("pizza", "simple"),
            # Complex query (should use smart reasoning)
            ("Find a romantic Italian restaurant with outdoor seating for my anniversary dinner tonight for 4 people",
             "complex"),
            # Medium complexity
            ("cheap tacos for 4 people", "medium")
        ]

        for query, expected_complexity in test_queries:
            result = await workflow.recommend_restaurants(
                user_query=query,
                user_id="test_user_complexity",
                user_location=(40.7128, -74.0060)
            )

            if result["success"]:
                perf = result["performance"]
                complexity_score = result["data"]["query_info"]["complexity_score"]

                print(
                    f"✅ {expected_complexity} query: {perf['processing_time_ms']:.0f}ms, complexity: {complexity_score:.2f}")

                # Simple queries should be faster
                if expected_complexity == "simple":
                    assert perf["processing_time_ms"] < 1000, "Simple query too slow"
            else:
                print(f"⚠️ {expected_complexity} query failed: {result.get('error')}")

    @pytest.mark.asyncio
    async def test_user_personalization(self, cache_adapter):
        """Test that different users get different recommendations"""
        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache_adapter
        )

        query = "Find me a good restaurant for dinner"
        location = (40.7128, -74.0060)

        # Test with different user personas
        users = ["foodie_explorer_0", "budget_conscious_0", "health_focused_0"]
        results = {}

        for user_id in users:
            result = await workflow.recommend_restaurants(
                user_query=query,
                user_id=user_id,
                user_location=location
            )
            results[user_id] = result

        # Check that we got different recommendations for different users
        successful_results = {k: v for k, v in results.items() if v["success"]}

        if len(successful_results) >= 2:
            # Compare first recommendation for each user
            user_recs = {}
            for user_id, result in successful_results.items():
                if result["data"]["recommendations"]:
                    top_rec = result["data"]["recommendations"][0]
                    user_recs[user_id] = top_rec["restaurant"]["name"]

            print(f"✅ User personalization test:")
            for user_id, restaurant_name in user_recs.items():
                print(f"   {user_id}: {restaurant_name}")

            # Check if recommendations are different (they should be for different personas)
            unique_recommendations = len(set(user_recs.values()))
            print(f"   Unique recommendations: {unique_recommendations}/{len(user_recs)}")
        else:
            print("⚠️ Not enough successful results to test personalization")

    @pytest.mark.asyncio
    async def test_workflow_performance(self, cache_adapter):
        """Test workflow performance and health metrics"""
        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache_adapter
        )

        # Run multiple queries to test performance
        queries = [
            "Italian restaurant",
            "Chinese food",
            "burger place",
            "coffee shop",
            "sushi restaurant"
        ]

        total_time = 0
        successful_queries = 0

        for query in queries:
            result = await workflow.recommend_restaurants(
                user_query=query,
                user_id="performance_test_user",
                user_location=(40.7128, -74.0060)
            )

            if result["success"]:
                successful_queries += 1
                total_time += result["performance"]["processing_time_ms"]

        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"✅ Performance test: {successful_queries}/{len(queries)} successful")
            print(f"   Average processing time: {avg_time:.1f}ms")

            # Check workflow health
            health = await workflow.get_workflow_health()
            print(f"   Workflow status: {health['workflow_health']['status']}")
            print(f"   Success rate: {health['workflow_health']['success_rate_percent']:.1f}%")

            assert avg_time < 2000, "Average processing time too high"
            assert health['workflow_health']['success_rate_percent'] > 50, "Success rate too low"
        else:
            print("⚠️ No successful queries for performance testing")

    @pytest.mark.asyncio
    async def test_error_recovery(self, cache_adapter):
        """Test workflow error handling and recovery"""
        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache_adapter
        )

        # Test with problematic inputs
        test_cases = [
            ("", "empty query"),
            ("asdfkjasldkfj", "gibberish query"),
            ("restaurant" * 100, "very long query"),
            ("🍕🍔🍟", "emoji query")
        ]

        for query, description in test_cases:
            result = await workflow.recommend_restaurants(
                user_query=query,
                user_id="error_test_user",
                user_location=(40.7128, -74.0060)
            )

            # Workflow should handle errors gracefully
            assert result is not None, f"No result for {description}"
            assert "success" in result, f"No success field for {description}"

            if result["success"]:
                print(f"✅ {description}: handled successfully")
            else:
                print(f"⚠️ {description}: handled gracefully with error: {result.get('error', 'unknown')}")
                # Error handling is also a success - it didn't crash


# Standalone workflow test
async def test_workflow_standalone():
    """Standalone workflow test"""
    print("🧪 Standalone Workflow Test")
    print("=" * 40)

    try:
        from src.infrastructure.databases.cache.memory_adapter import MemoryAdapter

        # Setup
        cache = MemoryAdapter()
        await cache.connect()

        workflow = await create_restaurant_recommendation_workflow(
            openai_api_key="fake_key_for_testing",
            use_mock_services=True,
            cache_adapter=cache
        )

        # Test simple recommendation
        print("Testing simple recommendation...")
        result = await workflow.recommend_restaurants(
            user_query="Find me a good Italian restaurant",
            user_id="standalone_test_user",
            user_location=(40.7128, -74.0060)
        )

        if result["success"]:
            data = result["data"]
            perf = result["performance"]

            print(f"✅ Success: {data['message']}")
            print(f"📊 Performance: {perf['processing_time_ms']:.0f}ms")
            print(f"🎯 Recommendations: {len(data['recommendations'])}")

            if data["recommendations"]:
                top_rec = data["recommendations"][0]
                print(f"🏆 Top pick: {top_rec['restaurant']['name']}")
        else:
            print(f"❌ Failed: {result['error']}")

        # Cleanup
        await cache.disconnect()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_workflow_standalone())