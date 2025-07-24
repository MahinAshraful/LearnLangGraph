from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime

from ...domain.models.restaurant import Restaurant
from ...domain.models.user import User, UserPreferences
from ...domain.models.query import ParsedQuery, QueryContext
from ...domain.models.recommendation import Recommendation, RecommendationContext, ScoreBreakdown
from ...domain.models.common import Location


class RecommendationState(TypedDict):
    """State that flows through the LangGraph workflow"""

    # Input data
    user_query: str
    user_id: str
    user_location: Optional[Location]

    # Parsed data
    parsed_query: Optional[ParsedQuery]
    query_context: Optional[QueryContext]
    user_preferences: Optional[UserPreferences]

    # Retrieved data
    nearby_restaurants: List[Restaurant]
    similar_users: List[UserPreferences]
    collaborative_restaurants: List[str]  # Restaurant IDs liked by similar users

    # Processed data
    candidate_restaurants: List[Restaurant]
    scored_recommendations: List[Recommendation]
    final_recommendations: List[Recommendation]

    # Workflow control
    should_use_smart_reasoning: bool
    complexity_score: float

    # Performance tracking
    processing_start_time: float
    api_calls_made: int
    cache_hits: int
    tokens_used: int

    # Error handling
    errors: List[str]
    warnings: List[str]

    # Final output
    response_message: str
    confidence_score: float