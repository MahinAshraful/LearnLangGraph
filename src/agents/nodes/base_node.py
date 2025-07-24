import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..state.recommendation_state import RecommendationState
from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all LangGraph agent nodes"""

    def __init__(self, name: str):
        self.name = name
        self.settings = get_settings()
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0

    async def __call__(self, state: RecommendationState) -> Dict[str, Any]:
        """Main entry point for node execution"""

        start_time = time.time()
        self.execution_count += 1

        try:
            logger.debug(f"Executing node: {self.name}")

            # Execute the node logic
            result = await self.execute(state)

            # Track performance
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            logger.debug(f"Node {self.name} completed in {execution_time:.3f}s")

            return result

        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            logger.error(f"Node {self.name} failed after {execution_time:.3f}s: {e}")

            # Return error state
            return {
                "errors": state.get("errors", []) + [f"{self.name}: {str(e)}"]
            }

    @abstractmethod
    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Execute the node logic and return state updates"""
        pass

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this node"""
        avg_time = self.total_execution_time / max(self.execution_count, 1)

        return {
            "node_name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": round(self.total_execution_time, 3),
            "average_execution_time": round(avg_time, 3),
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(self.execution_count, 1), 3)
        }

    def _update_performance_tracking(self, state: RecommendationState,
                                     api_calls: int = 0,
                                     cache_hits: int = 0,
                                     tokens_used: int = 0) -> Dict[str, Any]:
        """Helper to update performance tracking in state"""
        return {
            "api_calls_made": state.get("api_calls_made", 0) + api_calls,
            "cache_hits": state.get("cache_hits", 0) + cache_hits,
            "tokens_used": state.get("tokens_used", 0) + tokens_used
        }

    def _handle_error(self, state: RecommendationState, error: str,
                      is_fatal: bool = False) -> Dict[str, Any]:
        """Helper to handle errors consistently"""
        if is_fatal:
            return {
                "errors": state.get("errors", []) + [f"FATAL - {self.name}: {error}"]
            }
        else:
            return {
                "warnings": state.get("warnings", []) + [f"{self.name}: {error}"]
            }