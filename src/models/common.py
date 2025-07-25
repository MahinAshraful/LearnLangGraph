from datetime import datetime, time
from typing import Any, Dict, Optional, List
from pydantic import BaseModel as PydanticBaseModel, Field, validator
from enum import Enum
import uuid


class BaseModel(PydanticBaseModel):
    """Base model with common functionality"""

    model_config = {
        "validate_assignment": True,
        "use_enum_values": True,
        "populate_by_name": True  # renamed from allow_population_by_field_name
    }


class Location(BaseModel):
    """Geographic location with coordinates and address info"""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    address: Optional[str] = Field(None, description="Human-readable address")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State/Province")
    country: Optional[str] = Field(None, description="Country")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    neighborhood: Optional[str] = Field(None, description="Neighborhood or district")

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return coordinates as (lat, lng) tuple"""
        return (self.latitude, self.longitude)

    def distance_to(self, other: 'Location') -> float:
        """Calculate approximate distance to another location in kilometers"""
        import math

        # Haversine formula for great circle distance
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        # Radius of earth in kilometers
        r = 6371
        return c * r


class TimeSlot(BaseModel):
    """Represents a time period"""

    start_time: time = Field(..., description="Start time")
    end_time: time = Field(..., description="End time")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Monday)")

    @validator('end_time')
    def end_after_start(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v

    def is_open_at(self, check_time: time) -> bool:
        """Check if this time slot covers the given time"""
        return self.start_time <= check_time <= self.end_time


class WeatherCondition(str, Enum):
    """Weather conditions that might affect recommendations"""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    STORMY = "stormy"
    FOGGY = "foggy"


class Urgency(str, Enum):
    """Request urgency levels"""
    NOW = "now"  # Within next hour
    SOON = "soon"  # Within next few hours
    TODAY = "today"  # Later today
    THIS_WEEK = "this_week"  # This week
    PLANNING = "planning"  # Future planning


class DistancePreference(str, Enum):
    """Distance preference for recommendations"""
    WALKING = "walking"  # ~15 minutes walk
    NEARBY = "nearby"  # ~30 minutes travel
    CITY_WIDE = "city_wide"  # Anywhere in the city
    NO_PREFERENCE = "no_preference"


class EntityId(BaseModel):
    """Typed entity identifier"""

    value: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, EntityId):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False


class Score(BaseModel):
    """Represents a score with components for transparency"""

    total: float = Field(..., ge=0, le=1, description="Total score")
    components: Dict[str, float] = Field(description="Score breakdown")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence in score")
    reasoning: Optional[str] = Field(None, description="Human-readable reasoning")

    def __init__(self, **data):
        # Set default for components if not provided
        if 'components' not in data:
            data['components'] = {}
        super().__init__(**data)

    def add_component(self, name: str, value: float, weight: float = 1.0) -> None:
        """Add a score component"""
        self.components[name] = value
        # Recalculate total as weighted average
        if self.components:
            weighted_sum = sum(score * weight for score in self.components.values())
            total_weight = len(self.components) * weight
            self.total = min(weighted_sum / total_weight, 1.0)


class Feedback(BaseModel):
    """User feedback on recommendations"""

    user_id: str
    recommendation_id: str
    feedback_type: str = Field(..., pattern="^(like|dislike|clicked|booked|visited)$")
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None
    timestamp: datetime = Field(description="Timestamp of feedback")
    metadata: Dict[str, Any] = Field(description="Additional metadata")

    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        if 'metadata' not in data:
            data['metadata'] = {}
        super().__init__(**data)


class CacheKey(BaseModel):
    """Standardized cache key structure"""

    prefix: str
    identifier: str
    version: str = "v1"

    def __str__(self) -> str:
        return f"{self.prefix}:{self.version}:{self.identifier}"

    @classmethod
    def for_user_preferences(cls, user_id: str) -> 'CacheKey':
        return cls(prefix="user_prefs", identifier=user_id)

    @classmethod
    def for_restaurant_data(cls, place_id: str) -> 'CacheKey':
        return cls(prefix="restaurant", identifier=place_id)

    @classmethod
    def for_query_result(cls, query_hash: str) -> 'CacheKey':
        return cls(prefix="query_result", identifier=query_hash)