from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from enum import Enum
from pydantic import Field, validator

from .common import BaseModel, Location, EntityId
from .restaurant import RestaurantCategory, PriceLevel


class DietaryRestriction(str, Enum):
    """Dietary restrictions and preferences"""

    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    HALAL = "halal"
    KOSHER = "kosher"
    KETO = "keto"
    PALEO = "paleo"
    LOW_CARB = "low_carb"
    LOW_SODIUM = "low_sodium"


class AmbiancePreference(str, Enum):
    """Preferred restaurant ambiance"""

    ROMANTIC = "romantic"
    CASUAL = "casual"
    UPSCALE = "upscale"
    FAMILY_FRIENDLY = "family_friendly"
    BUSINESS = "business"
    TRENDY = "trendy"
    QUIET = "quiet"
    LIVELY = "lively"
    COZY = "cozy"
    OUTDOOR = "outdoor"
    AUTHENTIC = "authentic"
    MODERN = "modern"
    TRADITIONAL = "traditional"


class ActivityType(str, Enum):
    """Types of activities user has done"""

    VISITED = "visited"
    LIKED = "liked"
    DISLIKED = "disliked"
    BOOKMARKED = "bookmarked"
    REVIEWED = "reviewed"
    RECOMMENDED = "recommended"
    SHARED = "shared"


class UserPreferences(BaseModel):
    """User's dining preferences and taste profile"""

    user_id: str = Field(..., description="User identifier")

    # Cuisine preferences with weights
    favorite_cuisines: List[RestaurantCategory] = Field(description="Favorite cuisines")
    disliked_cuisines: List[RestaurantCategory] = Field(description="Disliked cuisines")
    cuisine_weights: Dict[str, float] = Field(description="Cuisine preference strength")

    # Price preferences
    preferred_price_levels: List[PriceLevel] = Field(description="Preferred price levels")
    budget_conscious: bool = False
    splurge_occasions: List[str] = Field(description="When user splurges")

    # Dietary needs
    dietary_restrictions: List[DietaryRestriction] = Field(description="Dietary restrictions")
    allergies: List[str] = Field(description="Allergies")

    # Ambiance and experience
    preferred_ambiance: List[AmbiancePreference] = Field(description="Preferred ambiance")
    group_size_preference: Optional[int] = Field(None, ge=1, le=20)

    # Location preferences
    home_location: Optional[Location] = None
    work_location: Optional[Location] = None
    preferred_neighborhoods: List[str] = Field(description="Preferred neighborhoods")
    max_travel_distance: float = Field(default=10.0, ge=0.1, le=100,
                                       description="Max travel distance in km")

    # Timing preferences
    preferred_meal_times: Dict[str, str] = Field(description="Meal time preferences")
    weekday_vs_weekend_preferences: Dict[str, Any] = Field(description="Day preferences")

    # Feature preferences
    must_have_features: List[str] = Field(description="Must have features")
    nice_to_have_features: List[str] = Field(description="Nice to have features")

    # Social aspects
    dining_companions: List[str] = Field(description="Frequent dining companions")
    occasion_preferences: Dict[str, List[str]] = Field(description="Occasion preferences")

    # Metadata
    last_updated: datetime = Field(description="Last updated timestamp")
    confidence_score: float = Field(default=0.0, ge=0, le=1,
                                    description="Confidence in preferences")

    def __init__(self, **data):
        if 'favorite_cuisines' not in data:
            data['favorite_cuisines'] = []
        if 'disliked_cuisines' not in data:
            data['disliked_cuisines'] = []
        if 'cuisine_weights' not in data:
            data['cuisine_weights'] = {}
        if 'preferred_price_levels' not in data:
            data['preferred_price_levels'] = []
        if 'splurge_occasions' not in data:
            data['splurge_occasions'] = []
        if 'dietary_restrictions' not in data:
            data['dietary_restrictions'] = []
        if 'allergies' not in data:
            data['allergies'] = []
        if 'preferred_ambiance' not in data:
            data['preferred_ambiance'] = []
        if 'preferred_neighborhoods' not in data:
            data['preferred_neighborhoods'] = []
        if 'preferred_meal_times' not in data:
            data['preferred_meal_times'] = {}
        if 'weekday_vs_weekend_preferences' not in data:
            data['weekday_vs_weekend_preferences'] = {}
        if 'must_have_features' not in data:
            data['must_have_features'] = []
        if 'nice_to_have_features' not in data:
            data['nice_to_have_features'] = []
        if 'dining_companions' not in data:
            data['dining_companions'] = []
        if 'occasion_preferences' not in data:
            data['occasion_preferences'] = {}
        if 'last_updated' not in data:
            data['last_updated'] = datetime.utcnow()
        super().__init__(**data)

    def add_cuisine_preference(self, cuisine: RestaurantCategory, weight: float = 1.0):
        """Add or update cuisine preference"""
        if cuisine not in self.favorite_cuisines:
            self.favorite_cuisines.append(cuisine)
        self.cuisine_weights[cuisine.value] = weight
        self.last_updated = datetime.utcnow()

    def get_cuisine_weight(self, cuisine: RestaurantCategory) -> float:
        """Get preference weight for a cuisine"""
        return self.cuisine_weights.get(cuisine.value, 0.0)

    def is_dietary_compatible(self, restaurant_features: Dict[str, Any]) -> bool:
        """Check if restaurant is compatible with dietary restrictions"""
        # Simplified check - in production would be more sophisticated
        for restriction in self.dietary_restrictions:
            if restriction == DietaryRestriction.VEGETARIAN:
                if not restaurant_features.get("vegetarian_options", False):
                    return False
            elif restriction == DietaryRestriction.VEGAN:
                if not restaurant_features.get("vegan_options", False):
                    return False
        return True


class UserActivity(BaseModel):
    """Record of user's activity with restaurants"""

    id: EntityId = Field(description="Activity ID")
    user_id: str = Field(..., description="User identifier")
    restaurant_id: str = Field(..., description="Restaurant identifier")
    place_id: str = Field(..., description="Google Places ID")

    activity_type: ActivityType = Field(..., description="Type of activity")
    rating: Optional[int] = Field(None, ge=1, le=5, description="User rating")
    review_text: Optional[str] = None

    # Context
    visit_date: Optional[datetime] = None
    party_size: int = Field(default=1, ge=1, le=20)
    occasion: Optional[str] = None
    dining_companions: List[str] = Field(description="Dining companions")

    # Experience details
    wait_time: Optional[int] = Field(None, ge=0, description="Wait time in minutes")
    service_rating: Optional[int] = Field(None, ge=1, le=5)
    food_rating: Optional[int] = Field(None, ge=1, le=5)
    ambiance_rating: Optional[int] = Field(None, ge=1, le=5)
    value_rating: Optional[int] = Field(None, ge=1, le=5)

    # Photos and content
    photos: List[str] = Field(description="Photo URLs")
    tags: List[str] = Field(description="Activity tags")

    # Metadata
    created_at: datetime = Field(description="Creation timestamp")
    source: str = Field(default="app", description="How activity was recorded")

    def __init__(self, **data):
        if 'id' not in data:
            data['id'] = EntityId()
        if 'dining_companions' not in data:
            data['dining_companions'] = []
        if 'photos' not in data:
            data['photos'] = []
        if 'tags' not in data:
            data['tags'] = []
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)

    @property
    def is_positive(self) -> bool:
        """Check if this is positive activity"""
        positive_activities = {ActivityType.VISITED, ActivityType.LIKED,
                               ActivityType.BOOKMARKED, ActivityType.RECOMMENDED}
        if self.activity_type in positive_activities:
            return True
        if self.rating and self.rating >= 4:
            return True
        return False

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score for this activity"""
        base_scores = {
            ActivityType.LIKED: 1.0,
            ActivityType.VISITED: 0.8,
            ActivityType.BOOKMARKED: 0.6,
            ActivityType.RECOMMENDED: 0.9,
            ActivityType.REVIEWED: 0.7,
            ActivityType.SHARED: 0.5,
            ActivityType.DISLIKED: -1.0
        }

        score = base_scores.get(self.activity_type, 0.0)

        # Boost score based on rating
        if self.rating:
            rating_boost = (self.rating - 3) / 2  # -1 to +1 scale
            score += rating_boost * 0.5

        return max(-1.0, min(1.0, score))


class User(BaseModel):
    """Main user entity"""

    id: EntityId = Field(description="User entity ID")
    user_id: str = Field(..., description="External user identifier")

    # Basic info
    name: Optional[str] = None
    email: Optional[str] = None

    # Preferences and profile
    preferences: UserPreferences = Field(description="User preferences")

    # Activity history
    recent_activities: List[UserActivity] = Field(description="Recent activities")
    favorite_restaurants: Set[str] = Field(description="Favorite restaurant IDs")
    disliked_restaurants: Set[str] = Field(description="Disliked restaurant IDs")

    # Social connections (for future collaborative filtering)
    friends: Set[str] = Field(description="Friend user IDs")
    following: Set[str] = Field(description="Following user IDs")

    # Usage patterns
    total_recommendations_requested: int = Field(default=0)
    total_restaurants_visited: int = Field(default=0)
    last_active: datetime = Field(description="Last active timestamp")

    # Privacy settings
    public_profile: bool = Field(default=False)
    share_activity: bool = Field(default=True)

    # Metadata
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Update timestamp")

    def __init__(self, **data):
        if 'id' not in data:
            data['id'] = EntityId()
        if 'preferences' not in data:
            data['preferences'] = UserPreferences(user_id=data.get('user_id', ''))
        if 'recent_activities' not in data:
            data['recent_activities'] = []
        if 'favorite_restaurants' not in data:
            data['favorite_restaurants'] = set()
        if 'disliked_restaurants' not in data:
            data['disliked_restaurants'] = set()
        if 'friends' not in data:
            data['friends'] = set()
        if 'following' not in data:
            data['following'] = set()
        if 'last_active' not in data:
            data['last_active'] = datetime.utcnow()
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        if 'updated_at' not in data:
            data['updated_at'] = datetime.utcnow()
        super().__init__(**data)

        # Set user_id in preferences if not set
        if self.preferences.user_id == "":
            self.preferences.user_id = self.user_id

    def add_activity(self, activity: UserActivity):
        """Add new activity and update derived data"""
        self.recent_activities.append(activity)

        # Update favorites/dislikes
        if activity.activity_type == ActivityType.LIKED or (activity.rating and activity.rating >= 4):
            self.favorite_restaurants.add(activity.place_id)
            self.disliked_restaurants.discard(activity.place_id)
        elif activity.activity_type == ActivityType.DISLIKED or (activity.rating and activity.rating <= 2):
            self.disliked_restaurants.add(activity.place_id)
            self.favorite_restaurants.discard(activity.place_id)

        # Keep only recent activities (last 100)
        if len(self.recent_activities) > 100:
            self.recent_activities = self.recent_activities[-100:]

        self.last_active = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def get_cuisine_affinity(self, cuisine: RestaurantCategory) -> float:
        """Calculate user's affinity for a specific cuisine based on activity history"""
        if not self.recent_activities:
            return self.preferences.get_cuisine_weight(cuisine)

        # Count positive and negative interactions with this cuisine
        positive_score = 0.0
        negative_score = 0.0
        total_interactions = 0

        for activity in self.recent_activities:
            # Would need to look up restaurant cuisine from activity
            # For now, use preference weights
            if cuisine in self.preferences.favorite_cuisines:
                positive_score += activity.weighted_score if activity.weighted_score > 0 else 0
            elif cuisine in self.preferences.disliked_cuisines:
                negative_score += abs(activity.weighted_score) if activity.weighted_score < 0 else 0
            total_interactions += 1

        if total_interactions == 0:
            return self.preferences.get_cuisine_weight(cuisine)

        # Combine preference weights with activity-based score
        preference_weight = self.preferences.get_cuisine_weight(cuisine)
        activity_score = (positive_score - negative_score) / total_interactions

        # Weighted combination: 60% activity, 40% stated preferences
        return 0.6 * activity_score + 0.4 * preference_weight

    def get_price_affinity(self, price_level: PriceLevel) -> float:
        """Calculate user's affinity for a price level"""
        if price_level in self.preferences.preferred_price_levels:
            return 1.0

        # Calculate based on activity patterns
        price_activities = []
        for activity in self.recent_activities:
            if activity.is_positive:
                price_activities.append(activity)

        if not price_activities:
            return 0.5  # Neutral if no data

        # Simple heuristic: prefer similar price levels to past positive experiences
        return min(1.0, len(price_activities) / 10.0)

    def update_preferences_from_activity(self):
        """Update preferences based on recent activity patterns"""
        if not self.recent_activities:
            return

        # Analyze recent positive activities to infer preferences
        positive_activities = [a for a in self.recent_activities if a.is_positive]

        # Update confidence score based on amount of data
        self.preferences.confidence_score = min(1.0, len(positive_activities) / 20.0)

        # Update timestamps
        self.preferences.last_updated = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    @property
    def experience_level(self) -> str:
        """Classify user's experience level"""
        if self.total_restaurants_visited < 5:
            return "new"
        elif self.total_restaurants_visited < 20:
            return "casual"
        elif self.total_restaurants_visited < 50:
            return "experienced"
        else:
            return "expert"

    @property
    def activity_recency_score(self) -> float:
        """Score based on how recently user has been active"""
        if not self.last_active:
            return 0.0

        days_since_active = (datetime.utcnow() - self.last_active).days

        if days_since_active <= 1:
            return 1.0
        elif days_since_active <= 7:
            return 0.8
        elif days_since_active <= 30:
            return 0.5
        else:
            return 0.2


class UserEmbedding(BaseModel):
    """User preference embeddings for vector similarity (Chroma/Pinecone integration)"""

    user_id: str = Field(..., description="User identifier")
    embedding_vector: List[float] = Field(..., description="Dense vector representation")
    embedding_version: str = Field(default="v1", description="Embedding model version")

    # Metadata for embedding generation
    preferences_hash: str = Field(..., description="Hash of preferences used")
    activity_count: int = Field(default=0, description="Number of activities used")
    last_updated: datetime = Field(description="Last updated timestamp")

    # Embedding components breakdown (for debugging/transparency)
    cuisine_component: List[float] = Field(description="Cuisine component")
    price_component: List[float] = Field(description="Price component")
    ambiance_component: List[float] = Field(description="Ambiance component")
    location_component: List[float] = Field(description="Location component")

    def __init__(self, **data):
        if 'last_updated' not in data:
            data['last_updated'] = datetime.utcnow()
        if 'cuisine_component' not in data:
            data['cuisine_component'] = []
        if 'price_component' not in data:
            data['price_component'] = []
        if 'ambiance_component' not in data:
            data['ambiance_component'] = []
        if 'location_component' not in data:
            data['location_component'] = []
        super().__init__(**data)

    @validator('embedding_vector')
    def validate_embedding_length(cls, v):
        # OpenAI text-embedding-3-large can be 1536 or 3072 dimensions
        if len(v) not in [1536, 3072]:
            raise ValueError('Embedding vector must be 1536 or 3072 dimensions')
        return v

    def similarity_to(self, other: 'UserEmbedding') -> float:
        """Calculate cosine similarity to another user embedding"""
        import numpy as np

        vec1 = np.array(self.embedding_vector)
        vec2 = np.array(other.embedding_vector)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @classmethod
    def from_user_preferences(cls, user: User, embedding_service) -> 'UserEmbedding':
        """Generate embedding from user preferences (would use OpenAI in production)"""
        # This would be implemented by the embedding service
        # For now, return a placeholder
        return cls(
            user_id=user.user_id,
            embedding_vector=[0.0] * 1536,  # Placeholder
            preferences_hash="placeholder",
            activity_count=len(user.recent_activities)
        )