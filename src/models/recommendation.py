from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import Field, validator
import uuid

from .common import BaseModel, Location, Score, EntityId
from .restaurant import Restaurant
from .user import User, UserPreferences
from .query import ParsedQuery, QueryContext


class RecommendationStrategy(str, Enum):
    """Different recommendation strategies"""

    CONTENT_BASED = "content_based"  # Based on item features
    COLLABORATIVE = "collaborative"  # Based on similar users
    HYBRID = "hybrid"  # Combination approach
    POPULARITY = "popularity"  # Based on general popularity
    CONTEXTUAL = "contextual"  # Context-aware recommendations
    KNOWLEDGE_BASED = "knowledge_based"  # Rule-based recommendations


class RecommendationReason(str, Enum):
    """Reasons why a restaurant was recommended"""

    CUISINE_MATCH = "cuisine_match"
    PRICE_MATCH = "price_match"
    LOCATION_CONVENIENT = "location_convenient"
    HIGHLY_RATED = "highly_rated"
    POPULAR_CHOICE = "popular_choice"
    SIMILAR_USERS_LIKED = "similar_users_liked"
    MATCHES_OCCASION = "matches_occasion"
    HAS_REQUIRED_FEATURES = "has_required_features"
    GOOD_FOR_PARTY_SIZE = "good_for_party_size"
    DIETARY_COMPATIBLE = "dietary_compatible"
    TRENDING_NOW = "trending_now"
    HIDDEN_GEM = "hidden_gem"
    WEATHER_APPROPRIATE = "weather_appropriate"


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of recommendation scoring"""

    # Core scoring components (aligned with your document: 50% preference, 30% context, 15% quality, 5% boost)
    preference_score: float = Field(0.0, ge=0, le=1, description="User preference match (50%)")
    context_score: float = Field(0.0, ge=0, le=1, description="Context relevance (30%)")
    quality_score: float = Field(0.0, ge=0, le=1, description="Restaurant quality (15%)")
    boost_score: float = Field(0.0, ge=0, le=1, description="Special boosts (5%)")

    # Sub-components for transparency
    cuisine_match: float = Field(0.0, ge=0, le=1)
    price_match: float = Field(0.0, ge=0, le=1)
    location_score: float = Field(0.0, ge=0, le=1)
    rating_score: float = Field(0.0, ge=0, le=1)
    popularity_score: float = Field(0.0, ge=0, le=1)
    feature_match: float = Field(0.0, ge=0, le=1)

    # Collaborative filtering
    collaborative_score: float = Field(0.0, ge=0, le=1)
    similar_users_count: int = Field(0, ge=0)

    # Contextual factors
    time_appropriateness: float = Field(0.0, ge=0, le=1)
    weather_appropriateness: float = Field(0.0, ge=0, le=1)
    occasion_match: float = Field(0.0, ge=0, le=1)

    @property
    def total_score(self) -> float:
        """Calculate weighted total score"""
        return (
                self.preference_score * 0.50 +
                self.context_score * 0.30 +
                self.quality_score * 0.15 +
                self.boost_score * 0.05
        )

    def explain_scoring(self) -> str:
        """Generate human-readable explanation of scoring"""
        explanations = []

        if self.preference_score > 0.7:
            explanations.append("strongly matches your preferences")
        elif self.preference_score > 0.5:
            explanations.append("matches your preferences")

        if self.quality_score > 0.8:
            explanations.append("highly rated restaurant")
        elif self.quality_score > 0.6:
            explanations.append("well-rated restaurant")

        if self.context_score > 0.7:
            explanations.append("perfect for the occasion")
        elif self.context_score > 0.5:
            explanations.append("suitable for your needs")

        if self.collaborative_score > 0.6:
            explanations.append("liked by similar users")

        return " and ".join(explanations) if explanations else "good overall choice"


class RecommendationContext(BaseModel):
    """Context information for making recommendations"""

    # Query context
    parsed_query: ParsedQuery
    user_preferences: Optional[UserPreferences] = None
    user_location: Optional[Location] = None

    # Environmental context
    current_time: datetime = Field(default_factory=datetime.utcnow)
    weather_condition: Optional[str] = None
    local_events: List[Dict[str, Any]] = Field(default_factory=list)

    # Social context
    dining_companions: List[str] = Field(default_factory=list)
    special_occasion: Optional[str] = None

    # System context
    available_restaurants: List[Restaurant] = Field(default_factory=list)
    similar_users: List[UserPreferences] = Field(default_factory=list)

    # Constraints
    max_results: int = Field(default=10, ge=1, le=50)
    strategy: RecommendationStrategy = RecommendationStrategy.HYBRID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching/serialization"""
        return {
            "query_type": self.parsed_query.query_type,
            "cuisines": [c.value for c in self.parsed_query.cuisine_preferences],
            "price_levels": [p.value for p in self.parsed_query.price_preferences],
            "party_size": self.parsed_query.social_context.party_size,
            "location": self.user_location.dict() if self.user_location else None,
            "time": self.current_time.isoformat(),
            "weather": self.weather_condition,
            "occasion": self.special_occasion
        }


class Recommendation(BaseModel):
    """Individual restaurant recommendation"""

    id: EntityId = Field(default_factory=EntityId)
    restaurant: Restaurant
    score: ScoreBreakdown

    # Reasoning
    primary_reasons: List[RecommendationReason] = Field(default_factory=list)
    explanation: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in recommendation")

    # Ranking
    rank: int = Field(..., ge=1, description="Position in recommendation list")

    # Personalization
    personalized: bool = Field(default=True)
    novelty_score: float = Field(0.0, ge=0, le=1, description="How novel/diverse this recommendation is")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    strategy_used: RecommendationStrategy = RecommendationStrategy.HYBRID

    @property
    def total_score(self) -> float:
        """Get total recommendation score"""
        return self.score.total_score

    def add_reason(self, reason: RecommendationReason):
        """Add a recommendation reason"""
        if reason not in self.primary_reasons:
            self.primary_reasons.append(reason)

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "id": str(self.id),
            "restaurant": {
                "place_id": self.restaurant.place_id,
                "name": self.restaurant.name,
                "cuisine": self.restaurant.primary_category.value,
                "rating": self.restaurant.rating,
                "price_level": self.restaurant.price_level.value if self.restaurant.price_level else None,
                "address": self.restaurant.formatted_address,
                "phone": self.restaurant.phone_number,
                "website": self.restaurant.website,
                "photos": self.restaurant.photos[:3],  # Limit photos
                "location": {
                    "lat": self.restaurant.location.latitude,
                    "lng": self.restaurant.location.longitude
                },
                "features": self.restaurant.features.dict(),
                "opening_hours": self.restaurant.opening_hours.dict() if self.restaurant.opening_hours else None
            },
            "recommendation": {
                "rank": self.rank,
                "score": round(self.total_score, 3),
                "confidence": round(self.confidence, 3),
                "explanation": self.explanation,
                "reasons": [reason.value for reason in self.primary_reasons],
                "novelty": round(self.novelty_score, 3)
            },
            "score_breakdown": {
                "preference_match": round(self.score.preference_score, 3),
                "context_relevance": round(self.score.context_score, 3),
                "quality": round(self.score.quality_score, 3),
                "boost": round(self.score.boost_score, 3)
            }
        }


class RecommendationSet(BaseModel):
    """Set of recommendations for a query"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recommendations: List[Recommendation]

    # Query information
    query_context: RecommendationContext
    user_id: Optional[str] = None

    # Results metadata
    total_candidates: int = Field(0, description="Total restaurants considered")
    strategy_used: RecommendationStrategy = RecommendationStrategy.HYBRID

    # Quality metrics
    average_score: float = Field(0.0, ge=0, le=1)
    score_variance: float = Field(0.0, ge=0)
    diversity_score: float = Field(0.0, ge=0, le=1, description="How diverse the recommendations are")

    # Performance metrics
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    api_calls_made: int = 0
    tokens_used: int = 0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate quality metrics"""
        if not self.recommendations:
            return

        # Calculate average score
        scores = [rec.total_score for rec in self.recommendations]
        self.average_score = sum(scores) / len(scores)

        # Calculate score variance
        mean = self.average_score
        self.score_variance = sum((score - mean) ** 2 for score in scores) / len(scores)

        # Calculate diversity (simplified - based on cuisine variety)
        cuisines = set(rec.restaurant.primary_category for rec in self.recommendations)
        self.diversity_score = min(len(cuisines) / len(self.recommendations), 1.0)

    def get_top_n(self, n: int) -> List[Recommendation]:
        """Get top N recommendations"""
        return sorted(self.recommendations, key=lambda r: r.total_score, reverse=True)[:n]

    def filter_by_score(self, min_score: float) -> List[Recommendation]:
        """Filter recommendations by minimum score"""
        return [rec for rec in self.recommendations if rec.total_score >= min_score]

    def get_by_cuisine(self, cuisine: str) -> List[Recommendation]:
        """Get recommendations for specific cuisine"""
        return [rec for rec in self.recommendations
                if rec.restaurant.primary_category.value == cuisine]

    def add_recommendation(self, recommendation: Recommendation):
        """Add a recommendation and recalculate metrics"""
        # Set rank
        recommendation.rank = len(self.recommendations) + 1
        self.recommendations.append(recommendation)
        self._calculate_metrics()

    def rerank_by_score(self):
        """Rerank recommendations by total score"""
        self.recommendations.sort(key=lambda r: r.total_score, reverse=True)
        for i, rec in enumerate(self.recommendations, 1):
            rec.rank = i
        self._calculate_metrics()

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "id": self.id,
            "query": {
                "original_query": self.query_context.parsed_query.original_query,
                "query_type": self.query_context.parsed_query.query_type.value,
                "party_size": self.query_context.parsed_query.social_context.party_size,
                "cuisines": [c.value for c in self.query_context.parsed_query.cuisine_preferences],
                "price_levels": [p.value for p in self.query_context.parsed_query.price_preferences]
            },
            "recommendations": [rec.to_api_format() for rec in self.recommendations],
            "metadata": {
                "total_results": len(self.recommendations),
                "total_candidates": self.total_candidates,
                "average_score": round(self.average_score, 3),
                "diversity_score": round(self.diversity_score, 3),
                "strategy_used": self.strategy_used.value,
                "processing_time_ms": round(self.processing_time_ms, 2),
                "cache_hit": self.cache_hit,
                "created_at": self.created_at.isoformat()
            },
            "performance": {
                "api_calls_made": self.api_calls_made,
                "tokens_used": self.tokens_used,
                "cache_hit": self.cache_hit
            }
        }


class RecommendationFeedback(BaseModel):
    """Feedback on a specific recommendation"""

    id: EntityId = Field(default_factory=EntityId)
    recommendation_id: str
    user_id: str

    # Feedback type
    feedback_type: str = Field(..., pattern="^(view|click|like|dislike|book|visit|share)$")
    rating: Optional[int] = Field(None, ge=1, le=5, description="User rating")

    # Detailed feedback
    comment: Optional[str] = None
    specific_issues: List[str] = Field(default_factory=list)  # price, location, cuisine, etc.

    # Context
    feedback_time: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None

    # Implicit feedback
    time_spent_viewing: Optional[float] = Field(None, ge=0, description="Seconds spent viewing")
    clicked_through: bool = False

    def to_learning_signal(self) -> Dict[str, Any]:
        """Convert to format for preference learning"""
        signal_strength = {
            "view": 0.1,
            "click": 0.3,
            "like": 0.8,
            "dislike": -0.8,
            "book": 1.0,
            "visit": 1.0,
            "share": 0.6
        }

        return {
            "user_id": self.user_id,
            "signal_type": self.feedback_type,
            "signal_strength": signal_strength.get(self.feedback_type, 0.0),
            "rating": self.rating,
            "timestamp": self.feedback_time.isoformat(),
            "issues": self.specific_issues
        }


class ExperimentVariant(BaseModel):
    """A/B testing variant for recommendation algorithms"""

    variant_id: str
    name: str
    description: str

    # Algorithm parameters
    scoring_weights: Dict[str, float] = Field(default_factory=dict)
    strategy: RecommendationStrategy = RecommendationStrategy.HYBRID
    feature_flags: Dict[str, bool] = Field(default_factory=dict)

    # Experiment metadata
    traffic_percentage: float = Field(..., ge=0, le=100)
    start_date: datetime
    end_date: Optional[datetime] = None

    # Performance tracking
    conversion_rate: float = Field(default=0.0)
    average_rating: float = Field(default=0.0)
    user_satisfaction: float = Field(default=0.0)


class RecommendationExperiment(BaseModel):
    """A/B testing experiment for recommendations"""

    experiment_id: str
    name: str
    hypothesis: str

    # Variants
    control_variant: ExperimentVariant
    test_variants: List[ExperimentVariant]

    # Experiment configuration
    user_assignment: Dict[str, str] = Field(default_factory=dict)  # user_id -> variant_id
    success_metrics: List[str] = Field(default_factory=list)

    # Status
    is_active: bool = True
    start_date: datetime
    end_date: Optional[datetime] = None

    def assign_user_to_variant(self, user_id: str) -> str:
        """Assign user to experiment variant"""
        if user_id in self.user_assignment:
            return self.user_assignment[user_id]

        # Simple hash-based assignment for consistent user experience
        import hashlib
        hash_value = int(hashlib.md5(f"{self.experiment_id}_{user_id}".encode()).hexdigest()[:8], 16)
        percentage = (hash_value % 100) + 1

        # Assign based on traffic percentages
        cumulative = 0
        all_variants = [self.control_variant] + self.test_variants

        for variant in all_variants:
            cumulative += variant.traffic_percentage
            if percentage <= cumulative:
                self.user_assignment[user_id] = variant.variant_id
                return variant.variant_id

        # Default to control
        self.user_assignment[user_id] = self.control_variant.variant_id
        return self.control_variant.variant_id


class CachedRecommendation(BaseModel):
    """Cached recommendation result"""

    cache_key: str
    recommendation_set: RecommendationSet

    # Cache metadata
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    hit_count: int = Field(default=0)

    # Cache validation
    user_preferences_hash: Optional[str] = None
    context_hash: str

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at

    def increment_hit_count(self):
        """Increment cache hit counter"""
        self.hit_count += 1

    @classmethod
    def create_cache_key(cls, user_id: str, query: ParsedQuery, context: RecommendationContext) -> str:
        """Generate cache key for recommendation"""
        import hashlib

        # Create hash from key components
        key_components = [
            user_id,
            query.original_query,
            str(query.social_context.party_size),
            str(context.current_time.date()),  # Date only, not time
            context.weather_condition or "",
            str(sorted([c.value for c in query.cuisine_preferences])),
            str(sorted([p.value for p in query.price_preferences]))
        ]

        key_string = "|".join(key_components)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()

        return f"rec:{cache_key}"