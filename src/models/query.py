from datetime import datetime, time
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field, validator

from .common import BaseModel, Location, Urgency, DistancePreference
from .restaurant import RestaurantCategory, PriceLevel
from .user import DietaryRestriction, AmbiancePreference


class QueryType(str, Enum):
    """Types of restaurant queries based on user intent"""

    CUISINE_SPECIFIC = "cuisine_specific"  # "Italian food"
    LOCATION_BASED = "location_based"  # "restaurants near me"
    OCCASION_BASED = "occasion_based"  # "date night restaurant"
    FEATURE_BASED = "feature_based"  # "outdoor seating"
    PRICE_BASED = "price_based"  # "cheap eats"
    TIME_BASED = "time_based"  # "late night food"
    MOOD_BASED = "mood_based"  # "comfort food"
    SOCIAL_BASED = "social_based"  # "good for groups"
    DIETARY_BASED = "dietary_based"  # "vegan restaurants"
    EXPERIENCE_BASED = "experience_based"  # "fine dining"
    GENERAL = "general"  # "good restaurants"


class LocationPreference(BaseModel):
    """Specific location preferences for the query"""

    center_location: Optional[Location] = None
    preferred_neighborhoods: List[str] = Field(default_factory=list)
    distance_preference: DistancePreference = DistancePreference.NEARBY
    max_distance_km: float = Field(default=10.0, ge=0.1, le=100)
    avoid_areas: List[str] = Field(default_factory=list)

    # Transportation considerations
    has_car: bool = True
    prefers_public_transit: bool = False
    walking_only: bool = False


class TimePreference(BaseModel):
    """Time-related preferences for dining"""

    preferred_time: Optional[time] = None
    meal_type: Optional[str] = Field(None, regex="^(breakfast|brunch|lunch|dinner|late_night)$")
    urgency: Urgency = Urgency.PLANNING
    flexible_timing: bool = True
    avoid_wait_times: bool = False
    reservation_required: bool = False


class SocialContext(BaseModel):
    """Social context of the dining experience"""

    party_size: int = Field(default=2, ge=1, le=20)
    occasion: Optional[str] = None
    companion_types: List[str] = Field(default_factory=list)  # family, friends, business, date
    celebration: Optional[str] = None  # birthday, anniversary, etc.
    special_needs: List[str] = Field(default_factory=list)  # wheelchair, high chair, etc.


class ParsedQuery(BaseModel):
    """Structured representation of a parsed restaurant query"""

    # Original query
    original_query: str = Field(..., description="Original user query text")
    query_type: QueryType = Field(..., description="Classified query type")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Parsing confidence")

    # Core preferences extracted
    cuisine_preferences: List[RestaurantCategory] = Field(default_factory=list)
    price_preferences: List[PriceLevel] = Field(default_factory=list)
    dietary_requirements: List[DietaryRestriction] = Field(default_factory=list)
    ambiance_preferences: List[AmbiancePreference] = Field(default_factory=list)

    # Context
    location_preference: LocationPreference = Field(default_factory=LocationPreference)
    time_preference: TimePreference = Field(default_factory=TimePreference)
    social_context: SocialContext = Field(default_factory=SocialContext)

    # Features and requirements
    required_features: List[str] = Field(default_factory=list)  # outdoor_seating, live_music, etc.
    nice_to_have_features: List[str] = Field(default_factory=list)
    deal_breakers: List[str] = Field(default_factory=list)

    # Quality preferences
    min_rating: float = Field(default=0.0, ge=0, le=5)
    prefer_popular: bool = False
    prefer_hidden_gems: bool = False

    # Output preferences
    max_results: int = Field(default=10, ge=1, le=50)
    sort_preference: str = Field(default="relevance",
                                 regex="^(relevance|rating|distance|price|popularity)$")

    # Metadata
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    language: str = Field(default="en")
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    @validator('party_size')
    def validate_party_size(cls, v, values):
        if 'social_context' in values:
            # Ensure consistency between party_size and social_context
            return v
        return v

    def to_search_criteria(self) -> Dict[str, Any]:
        """Convert to restaurant search criteria"""
        return {
            "categories": self.cuisine_preferences,
            "price_levels": self.price_preferences,
            "min_rating": self.min_rating,
            "required_features": self.required_features,
            "max_distance_km": self.location_preference.max_distance_km,
            "party_size": self.social_context.party_size,
            "max_results": self.max_results
        }

    @property
    def complexity_score(self) -> float:
        """Calculate query complexity for routing decisions"""
        complexity = 0.0

        # Multiple cuisines increase complexity
        complexity += len(self.cuisine_preferences) * 0.1

        # Dietary restrictions add complexity
        complexity += len(self.dietary_requirements) * 0.2

        # Special features add complexity
        complexity += len(self.required_features) * 0.1

        # Social context complexity
        if self.social_context.party_size > 4:
            complexity += 0.2
        if self.social_context.occasion:
            complexity += 0.15

        # Time constraints
        if self.time_preference.urgency in [Urgency.NOW, Urgency.SOON]:
            complexity += 0.2

        # Ambiance preferences
        complexity += len(self.ambiance_preferences) * 0.1

        return min(complexity, 1.0)

    @property
    def requires_smart_reasoning(self) -> bool:
        """Determine if query needs advanced LLM reasoning"""
        return (
                self.complexity_score > 0.5 or
                len(self.required_features) > 2 or
                self.social_context.occasion is not None or
                len(self.ambiance_preferences) > 1 or
                self.time_preference.urgency in [Urgency.NOW, Urgency.SOON]
        )


class QueryIntent(BaseModel):
    """High-level intent classification for query routing"""

    primary_intent: QueryType
    secondary_intents: List[QueryType] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)

    # Intent-specific data
    entities: Dict[str, Any] = Field(default_factory=dict)
    sentiment: str = Field(default="neutral", regex="^(positive|neutral|negative)$")

    # Processing hints
    use_cache: bool = True
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")

    @classmethod
    def from_query_text(cls, query_text: str) -> 'QueryIntent':
        """Quick intent classification (would use LLM in production)"""
        # Simplified rule-based classification for demo
        query_lower = query_text.lower()

        # Cuisine-specific queries
        cuisines = ["italian", "chinese", "japanese", "mexican", "thai", "indian", "french"]
        if any(cuisine in query_lower for cuisine in cuisines):
            return cls(primary_intent=QueryType.CUISINE_SPECIFIC, confidence=0.9)

        # Location-based queries
        location_words = ["near", "nearby", "close", "around", "area"]
        if any(word in query_lower for word in location_words):
            return cls(primary_intent=QueryType.LOCATION_BASED, confidence=0.8)

        # Price-based queries
        price_words = ["cheap", "budget", "expensive", "fine dining", "affordable"]
        if any(word in query_lower for word in price_words):
            return cls(primary_intent=QueryType.PRICE_BASED, confidence=0.8)

        # Feature-based queries
        feature_words = ["outdoor", "live music", "parking", "delivery", "takeout"]
        if any(word in query_lower for word in feature_words):
            return cls(primary_intent=QueryType.FEATURE_BASED, confidence=0.7)

        # Default to general
        return cls(primary_intent=QueryType.GENERAL, confidence=0.5)


class QueryContext(BaseModel):
    """Additional context for query processing"""

    user_id: str
    session_id: str
    device_type: str = Field(default="web", regex="^(web|mobile|tablet|voice)$")

    # User context
    user_location: Optional[Location] = None
    time_of_query: datetime = Field(default_factory=datetime.utcnow)

    # Conversation context
    previous_queries: List[str] = Field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

    # External context
    weather_condition: Optional[str] = None
    local_events: List[Dict[str, Any]] = Field(default_factory=list)

    # Personalization hints
    use_personalization: bool = True
    include_social_signals: bool = True

    def add_to_history(self, query: str, result_count: int):
        """Add query to conversation history"""
        self.previous_queries.append(query)
        self.conversation_history.append({
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "result_count": result_count
        })

        # Keep only recent history
        if len(self.previous_queries) > 10:
            self.previous_queries = self.previous_queries[-10:]
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]


class QueryResult(BaseModel):
    """Results from query processing"""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parsed_query: ParsedQuery
    query_context: QueryContext

    # Processing metadata
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    confidence: float = Field(..., ge=0, le=1)

    # Results
    restaurant_ids: List[str] = Field(default_factory=list)
    total_results: int = 0

    # Reasoning
    reasoning_used: bool = False
    explanation: Optional[str] = None

    # Performance tracking
    api_calls_made: int = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0

    created_at: datetime = Field(default_factory=datetime.utcnow)