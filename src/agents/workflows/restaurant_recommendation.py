import time
import logging
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END

from ..recommendation_state import RecommendationState
from ..nodes.query_parser import QueryParserNode
from ..nodes.user_context import UserContextNode
from ..nodes.data_retrieval import DataRetrievalNode
from ..nodes.candidate_filter import CandidateFilterNode
from ..nodes.scoring import ScoringNode
from ..nodes.output_formatter import OutputFormatterNode
from ...infrastructure.api_clients.openai.client import OpenAIClient
from ...infrastructure.api_clients.google_places.client import GooglePlacesClient
from ...infrastructure.databases.vector_db.base import VectorDBAdapter
from ...infrastructure.api_clients.places_factory import PlacesClientFactory
from ...config.settings import get_settings


logger = logging.getLogger(__name__)


class RestaurantRecommendationWorkflow:
    """Main workflow orchestrating restaurant recommendations using LangGraph"""

    def __init__(self,
                 openai_client: OpenAIClient,
                 places_client,
                 vector_db: VectorDBAdapter):

        self.settings = get_settings()

        # Initialize agents
        self.query_parser = QueryParserNode(openai_client)
        self.user_context = UserContextNode(vector_db)
        self.data_retrieval = DataRetrievalNode(places_client)
        self.candidate_filter = CandidateFilterNode()
        self.scoring = ScoringNode()
        self.output_formatter = OutputFormatterNode()

        # Build the workflow graph
        self.graph = self._build_workflow_graph()

        # Performance tracking
        self.total_executions = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create state graph
        graph_builder = StateGraph(RecommendationState)

        # Add all nodes
        graph_builder.add_node("query_parser", self.query_parser)
        graph_builder.add_node("user_context", self.user_context)
        graph_builder.add_node("data_retrieval", self.data_retrieval)
        graph_builder.add_node("candidate_filter", self.candidate_filter)
        graph_builder.add_node("scoring", self.scoring)
        graph_builder.add_node("output_formatter", self.output_formatter)

        # Define the workflow edges
        graph_builder.add_edge(START, "query_parser")
        graph_builder.add_edge("query_parser", "user_context")
        graph_builder.add_edge("user_context", "data_retrieval")
        graph_builder.add_edge("data_retrieval", "candidate_filter")
        graph_builder.add_edge("candidate_filter", "scoring")

        # Conditional routing - skip smart reasoning for simple queries
        graph_builder.add_conditional_edges(
            "scoring",
            self._should_use_smart_reasoning,
            {
                "output": "output_formatter"  # ← Remove smart_reasoning, go directly to output
            }
        )

        graph_builder.add_edge("output_formatter", END)

        return graph_builder.compile()

    def _should_use_smart_reasoning(self, state: RecommendationState) -> str:
        """Decide whether to use smart reasoning (GPT-4) or go directly to output"""

        # For MVP, always go directly to output
        # In production, would check complexity_score and other factors
        return "output"  # Always return "output"

    async def recommend_restaurants(self,
                                    user_query: str,
                                    user_id: str,
                                    user_location: Optional[tuple] = None) -> Dict[str, Any]:
        """Main entry point for restaurant recommendations"""

        start_time = time.time()
        self.total_executions += 1

        try:
            # Initialize state
            from ...models.common import Location

            location = None
            if user_location:
                location = Location(
                    latitude=user_location[0],
                    longitude=user_location[1]
                )

            initial_state = RecommendationState(
                user_query=user_query,
                user_id=user_id,
                user_location=location,
                processing_start_time=start_time,
                api_calls_made=0,
                cache_hits=0,
                tokens_used=0,
                errors=[],
                warnings=[],
                response_message="",
                confidence_score=0.0,
                nearby_restaurants=[],
                similar_users=[],
                collaborative_restaurants=[],
                candidate_restaurants=[],
                scored_recommendations=[],
                final_recommendations=[],
                should_use_smart_reasoning=False,
                complexity_score=0.0,
                parsed_query=None,
                query_context=None,
                user_preferences=None
            )

            print(f"DEBUG WORKFLOW: Initial state type: {type(initial_state)}")
            print(f"DEBUG WORKFLOW: Initial state keys: {list(initial_state.keys())}")
            print(f"DEBUG WORKFLOW: user_query in state: {initial_state.get('user_query')}")

            logger.info(f"Starting recommendation workflow for user {user_id}: '{user_query}'")

            # Execute the workflow
            print("DEBUG WORKFLOW: About to call graph.ainvoke")
            final_state = await self.graph.ainvoke(initial_state)
            print(f"DEBUG WORKFLOW: Final state type: {type(final_state)}")
            print(
                f"DEBUG WORKFLOW: Final state keys: {list(final_state.keys()) if hasattr(final_state, 'keys') else 'No keys method'}")

            # Calculate total processing time
            total_time = time.time() - start_time
            self.total_processing_time += total_time

            # Check for fatal errors
            if final_state.get("errors"):
                self.error_count += 1
                logger.error(f"Workflow failed with errors: {final_state['errors']}")

                return {
                    "success": False,
                    "error": final_state["errors"][0],
                    "data": {
                        "message": "Sorry, I encountered an error processing your request. Please try again.",
                        "recommendations": []
                    },
                    "performance": {
                        "processing_time_ms": total_time * 1000,
                        "api_calls": final_state.get("api_calls_made", 0),
                        "cache_hits": final_state.get("cache_hits", 0),
                        "tokens_used": final_state.get("tokens_used", 0)
                    }
                }

            # Success case
            self.success_count += 1

            recommendations = final_state.get("final_recommendations", [])
            formatted_recommendations = [
                self.output_formatter._format_recommendation_for_api(rec)
                for rec in recommendations
            ]

            logger.info(f"Workflow completed successfully in {total_time:.3f}s: {len(recommendations)} recommendations")

            return {
                "success": True,
                "data": {
                    "message": final_state.get("response_message", ""),
                    "recommendations": formatted_recommendations,
                    "query_info": {
                        "original_query": user_query,
                        "parsed_query_type": getattr(final_state.get("parsed_query"), "query_type", "general") if final_state.get("parsed_query") else "general",
                        "complexity_score": final_state.get("complexity_score", 0.0),
                        "confidence_score": final_state.get("confidence_score", 0.0)
                    }
                },
                "performance": {
                    "processing_time_ms": round(total_time * 1000, 2),
                    "api_calls": final_state.get("api_calls_made", 0),
                    "cache_hits": final_state.get("cache_hits", 0),
                    "tokens_used": final_state.get("tokens_used", 0),
                    "total_candidates": len(final_state.get("candidate_restaurants", [])),
                    "workflow_steps_completed": self._count_completed_steps(final_state)
                },
                "debug_info": {
                    "warnings": final_state.get("warnings", []),
                    "user_has_preferences": bool(final_state.get("user_preferences")),
                    "similar_users_found": len(final_state.get("similar_users", [])),
                    "collaborative_signals": len(final_state.get("collaborative_restaurants", []))
                } if self.settings.debug else {}
            }

        except Exception as e:
            self.error_count += 1
            total_time = time.time() - start_time

            logger.error(f"Workflow execution failed: {e}")
            import traceback
            print(f"DEBUG WORKFLOW: Exception details:")
            traceback.print_exc()

            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "data": {
                    "message": "Sorry, something went wrong while processing your request. Please try again.",
                    "recommendations": []
                },
                "performance": {
                    "processing_time_ms": round(total_time * 1000, 2),
                    "api_calls": 0,
                    "cache_hits": 0,
                    "tokens_used": 0
                }
            }

    def _count_completed_steps(self, final_state: RecommendationState) -> int:
        """Count how many workflow steps completed successfully"""

        steps = 0

        if final_state.get("parsed_query"):
            steps += 1
        if final_state.get("user_preferences") is not None:
            steps += 1
        if final_state.get("nearby_restaurants"):
            steps += 1
        if final_state.get("candidate_restaurants") is not None:
            steps += 1
        if final_state.get("scored_recommendations"):
            steps += 1
        if final_state.get("response_message"):
            steps += 1

        return steps

    async def get_workflow_health(self) -> Dict[str, Any]:
        """Get workflow health and performance statistics"""

        success_rate = (self.success_count / max(self.total_executions, 1)) * 100
        avg_processing_time = self.total_processing_time / max(self.total_executions, 1)

        # Get individual node performance
        node_stats = {
            "query_parser": self.query_parser.get_performance_stats(),
            "user_context": self.user_context.get_performance_stats(),
            "data_retrieval": self.data_retrieval.get_performance_stats(),
            "candidate_filter": self.candidate_filter.get_performance_stats(),
            "scoring": self.scoring.get_performance_stats(),
            "output_formatter": self.output_formatter.get_performance_stats()
        }

        return {
            "workflow_health": {
                "status": "healthy" if success_rate > 80 else "degraded" if success_rate > 50 else "unhealthy",
                "total_executions": self.total_executions,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate_percent": round(success_rate, 2),
                "average_processing_time_ms": round(avg_processing_time * 1000, 2)
            },
            "node_performance": node_stats
        }

    def reset_stats(self):
        """Reset performance statistics"""

        self.total_executions = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0

        # Reset individual node stats
        for node in [self.query_parser, self.user_context, self.data_retrieval,
                    self.candidate_filter, self.scoring, self.output_formatter]:
            node.execution_count = 0
            node.total_execution_time = 0.0
            node.error_count = 0


async def create_restaurant_recommendation_workflow(openai_api_key: str,
                                                    use_mock_services: bool = False,
                                                    cache_adapter=None) -> RestaurantRecommendationWorkflow:
    """Create and initialize the restaurant recommendation workflow with LLM-powered places client"""

    # Initialize OpenAI client
    if use_mock_services or not openai_api_key or openai_api_key == "fake_key_for_testing":
        from ...infrastructure.api_clients.openai.client import OpenAIClient
        openai_client = OpenAIClient(cache_adapter=cache_adapter, use_mock=True)
        print("DEBUG WORKFLOW: Using mock OpenAI client")
    else:
        from ...infrastructure.api_clients.openai.client import OpenAIClient
        openai_client = OpenAIClient(api_key=openai_api_key, cache_adapter=cache_adapter)
        print("DEBUG WORKFLOW: Using real OpenAI client")

    # Initialize Places client with OpenAI client for LLM cuisine classification
    if use_mock_services:
        places_client = PlacesClientFactory.create_client(
            "mock",
            cache_adapter=cache_adapter,
            openai_client=openai_client  # 🤖 Pass OpenAI client!
        )
    else:
        places_client = PlacesClientFactory.create_client(
            "auto",
            cache_adapter=cache_adapter,
            openai_client=openai_client  # 🤖 Pass OpenAI client!
        )

    # Initialize Vector DB (mock for MVP)
    if use_mock_services:
        from ...infrastructure.databases.vector_db.mock_adapter import MockVectorAdapter
        vector_db = MockVectorAdapter()
    else:
        # TODO: Implement ChromaAdapter later
        # from ...infrastructure.databases.vector_db.chroma_adapter import ChromaAdapter
        # vector_db = ChromaAdapter()

        # For now, fall back to mock even when use_mock_services=False
        from ...infrastructure.databases.vector_db.mock_adapter import MockVectorAdapter
        vector_db = MockVectorAdapter()
        print("WARNING: ChromaAdapter not implemented, using MockVectorAdapter")

    # Connect to services
    await vector_db.connect()

    # Create workflow with LLM-enabled places client
    workflow = RestaurantRecommendationWorkflow(
        openai_client=openai_client,
        places_client=places_client,  # Now has LLM cuisine classification!
        vector_db=vector_db
    )

    logger.info("Restaurant recommendation workflow created with LLM cuisine classification")
    return workflow


# Simple demo function for testing
async def demo_workflow():
    """Demo function to test the workflow"""

    import os
    from ...infrastructure.databases.cache.memory_adapter import MemoryAdapter

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return

    # Create cache
    cache = MemoryAdapter()
    await cache.connect()

    # Create workflow
    workflow = await create_restaurant_recommendation_workflow(
        openai_api_key=openai_api_key,
        use_mock_services=False,
        cache_adapter=cache
    )

    # Test queries
    test_queries = [
        {
            "query": "Find me a good Italian restaurant for dinner tonight",
            "user_id": "foodie_explorer_0",
            "location": (40.7128, -74.0060)  # NYC
        },
        {
            "query": "I want cheap tacos for 4 people",
            "user_id": "budget_conscious_0",
            "location": (40.7589, -73.9851)  # Upper West Side
        },
        {
            "query": "Looking for expensive Japanese restaurant for business dinner",
            "user_id": "business_diner_0",
            "location": (40.7505, -73.9934)  # Midtown
        },
        {
            "query": "I need a vegan-friendly Indian place",
            "user_id": "MAHINTEST",
            "location": (40.7505, -73.9934)  # Midtown
        }
    ]

    print("🍽️  Restaurant Recommendation Workflow Demo")
    print("=" * 60)

    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: '{test['query']}'")
        print(f"User: {test['user_id']}")

        # Get recommendations
        result = await workflow.recommend_restaurants(
            user_query=test["query"],
            user_id=test["user_id"],
            user_location=test["location"]
        )

        if result["success"]:
            data = result["data"]
            perf = result["performance"]

            print(f"\n✅ {data['message']}")
            print(f"\n📍 Top Recommendations:")

            recommendations = data.get("recommendations", [])
            if recommendations:
                for i, rec_dict in enumerate(recommendations[:3], 1):
                    restaurant = rec_dict["restaurant"]
                    recommendation = rec_dict["recommendation"]

                    print(f"  {i}. {restaurant['name']} ({restaurant['cuisine']})")
                    print(
                        f"     ⭐ {restaurant['rating']}/5 | {restaurant.get('price_level', 'Price varies')} | Score: {recommendation['score']}")
                    print(f"     💡 {recommendation['explanation']}")

            print(f"\n⚡ Performance:")
            print(f"  Processing Time: {perf['processing_time_ms']:.0f}ms")
            print(f"  API Calls: {perf['api_calls']}")
            print(f"  Cache Hits: {perf['cache_hits']}")
            print(f"  Workflow Steps: {perf['workflow_steps_completed']}/6")

        else:
            print(f"❌ Error: {result['error']}")

    # Show workflow health
    print(f"\n📊 Workflow Health:")
    health = await workflow.get_workflow_health()
    wf_health = health["workflow_health"]
    print(f"  Status: {wf_health['status']}")
    print(f"  Success Rate: {wf_health['success_rate_percent']:.1f}%")
    print(f"  Avg Processing Time: {wf_health['average_processing_time_ms']:.0f}ms")

    # Cleanup
    await cache.disconnect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_workflow())