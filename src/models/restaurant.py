from datetime import time, datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field, field_validator

from .common import BaseModel, Location, TimeSlot, EntityId


class RestaurantCategory(str, Enum):
    """Restaurant cuisine/category types - aligned with Google Places"""

    # Primary cuisines
    AMERICAN = "american"
    ITALIAN = "italian"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    THAI = "thai"
    INDIAN = "indian"
    MEXICAN = "mexican"
    FRENCH = "french"
    MEDITERRANEAN = "mediterranean"
    KOREAN = "korean"
    VIETNAMESE = "vietnamese"
    GREEK = "greek"
    SPANISH = "spanish"
    MIDDLE_EASTERN = "middle_eastern"

    # Food types
    PIZZA = "pizza"
    SUSHI = "sushi"
    BBQ = "bbq"
    SEAFOOD = "seafood"
    STEAKHOUSE = "steakhouse"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"

    # Service types
    FAST_FOOD = "fast_food"
    CASUAL_DINING = "casual_dining"
    FINE_DINING = "fine_dining"
    CAFE = "cafe"
    BAR = "bar"
    BAKERY = "bakery"


class PriceLevel(int, Enum):
    """Price levels following Google Places API standard"""

    INEXPENSIVE = 1  # $
    MODERATE = 2  # $$
    EXPENSIVE = 3  # $$$
    VERY_EXPENSIVE = 4  # $$$$

    @property
    def symbol(self) -> str:
        return "$" * self.value

    @property
    def description(self) -> str:
        descriptions = {
            1: "Budget-friendly",
            2: "Moderate pricing",
            3: "Upscale",
            4: "Fine dining"
        }
        return descriptions[self.value]


class OpeningHours(BaseModel):
    """Restaurant opening hours"""

    monday: Optional[List[TimeSlot]] = Field(description="Monday time slots")
    tuesday: Optional[List[TimeSlot]] = Field(description="Tuesday time slots")
    wednesday: Optional[List[TimeSlot]] = Field(description="Wednesday time slots")
    thursday: Optional[List[TimeSlot]] = Field(description="Thursday time slots")
    friday: Optional[List[TimeSlot]] = Field(description="Friday time slots")
    saturday: Optional[List[TimeSlot]] = Field(description="Saturday time slots")
    sunday: Optional[List[TimeSlot]] = Field(description="Sunday time slots")

    def __init__(self, **data):
        # Set default empty lists for each day if not provided
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day not in data:
                data[day] = []
        super().__init__(**data)

    def is_open_at(self, check_time: datetime) -> bool:
        """Check if restaurant is open at given datetime"""
        day_name = check_time.strftime("%A").lower()
        time_slots = getattr(self, day_name, [])

        check_time_only = check_time.time()
        return any(slot.is_open_at(check_time_only) for slot in time_slots)

    def next_opening(self, from_time: datetime) -> Optional[datetime]:
        """Find next opening time after given datetime"""
        # Simplified implementation - in production would handle week rollovers
        day_name = from_time.strftime("%A").lower()
        time_slots = getattr(self, day_name, [])

        current_time = from_time.time()
        for slot in time_slots:
            if slot.start_time > current_time:
                return datetime.combine(from_time.date(), slot.start_time)

        return None


class RestaurantFeatures(BaseModel):
    """Features and amenities"""

    outdoor_seating: bool = False
    live_music: bool = False
    wifi: bool = False
    parking_available: bool = False
    wheelchair_accessible: bool = False
    accepts_reservations: bool = False
    delivery_available: bool = False
    takeout_available: bool = False
    good_for_groups: bool = False
    good_for_kids: bool = False
    romantic: bool = False
    casual: bool = True
    upscale: bool = False
    has_bar: bool = False
    happy_hour: bool = False


class PopularityData(BaseModel):
    """Real-time popularity data (from Google Places 'popular times')"""

    current_popularity: Optional[int] = Field(None, ge=0, le=100,
                                              description="Current busy percentage")
    typical_wait_time: Optional[int] = Field(None, ge=0,
                                             description="Typical wait time in minutes")
    peak_hours: List[int] = Field(description="Peak hours (24-hour format)")
    is_usually_busy_now: Optional[bool] = None
    last_updated: datetime = Field(description="Last updated timestamp")

    def __init__(self, **data):
        if 'peak_hours' not in data:
            data['peak_hours'] = []
        if 'last_updated' not in data:
            data['last_updated'] = datetime.utcnow()
        super().__init__(**data)


class Restaurant(BaseModel):
    """Main restaurant entity - designed for Google Places integration"""

    # Core identification
    id: EntityId = Field(description="Unique entity ID")
    place_id: str = Field(..., description="Google Places ID")
    name: str = Field(..., min_length=1, description="Restaurant name")

    # Location
    location: Location = Field(..., description="Geographic location")

    # Categories and type
    primary_category: RestaurantCategory = Field(..., description="Primary cuisine type")
    secondary_categories: List[RestaurantCategory] = Field(description="Secondary categories")
    google_types: List[str] = Field(description="Google Places types")

    # Pricing and quality
    price_level: Optional[PriceLevel] = Field(None, description="Price level")
    rating: float = Field(0.0, ge=0, le=5, description="Average rating")
    user_ratings_total: int = Field(0, ge=0, description="Total number of ratings")

    # Contact and web presence
    phone_number: Optional[str] = None
    website: Optional[str] = None
    formatted_address: str = Field(..., description="Full formatted address")

    # Operational info
    opening_hours: Optional[OpeningHours] = None
    features: RestaurantFeatures = Field(description="Restaurant features")

    # Real-time data
    popularity: Optional[PopularityData] = None

    # Rich content
    photos: List[str] = Field(description="Photo URLs")
    reviews_summary: Optional[str] = Field(None, description="AI-generated review summary")

    # Metadata
    last_updated: datetime = Field(description="Last updated timestamp")
    data_source: str = Field(default="google_places", description="Data source")

    def __init__(self, **data):
        # Set defaults for mutable fields
        if 'id' not in data:
            data['id'] = EntityId()
        if 'secondary_categories' not in data:
            data['secondary_categories'] = []
        if 'google_types' not in data:
            data['google_types'] = []
        if 'features' not in data:
            data['features'] = RestaurantFeatures()
        if 'photos' not in data:
            data['photos'] = []
        if 'last_updated' not in data:
            data['last_updated'] = datetime.utcnow()
        super().__init__(**data)

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if v < 0 or v > 5:
            raise ValueError('Rating must be between 0 and 5')
        return v

    @property
    def is_open_now(self) -> bool:
        """Check if restaurant is currently open"""
        if not self.opening_hours:
            return True  # Assume open if no hours specified
        return self.opening_hours.is_open_at(datetime.now())

    @property
    def quality_score(self) -> float:
        """Calculate quality score based on rating and review count"""
        if self.user_ratings_total == 0:
            return 0.0

        # Weight rating by number of reviews (more reviews = more reliable)
        review_confidence = min(self.user_ratings_total / 100, 1.0)
        return (self.rating / 5.0) * review_confidence

    @property
    def popularity_score(self) -> float:
        """Current popularity score"""
        if not self.popularity or self.popularity.current_popularity is None:
            return 0.5  # Default neutral popularity
        return self.popularity.current_popularity / 100.0

    def matches_category(self, category: RestaurantCategory) -> bool:
        """Check if restaurant matches a specific category"""
        return (self.primary_category == category or
                category in self.secondary_categories)

    def distance_from(self, location: Location) -> float:
        """Calculate distance from given location in kilometers"""
        return self.location.distance_to(location)

    def to_google_places_format(self) -> Dict[str, Any]:
        """Convert to Google Places API response format for compatibility"""
        return {
            "place_id": self.place_id,
            "name": self.name,
            "geometry": {
                "location": {
                    "lat": self.location.latitude,
                    "lng": self.location.longitude
                }
            },
            "rating": self.rating,
            "user_ratings_total": self.user_ratings_total,
            "price_level": self.price_level.value if self.price_level else None,
            "types": self.google_types,
            "formatted_address": self.formatted_address,
            "photos": [{"photo_reference": url} for url in self.photos]
        }


class RestaurantSearchCriteria(BaseModel):
    """Criteria for searching restaurants"""

    location: Location
    radius_km: float = Field(default=5.0, ge=0.1, le=50)
    categories: List[RestaurantCategory] = Field(description="Restaurant categories")
    min_rating: float = Field(default=0.0, ge=0, le=5)
    price_levels: List[PriceLevel] = Field(description="Price levels")
    must_be_open: bool = True
    required_features: List[str] = Field(description="Required features")
    max_results: int = Field(default=20, ge=1, le=100)

    def __init__(self, **data):
        if 'categories' not in data:
            data['categories'] = []
        if 'price_levels' not in data:
            data['price_levels'] = []
        if 'required_features' not in data:
            data['required_features'] = []
        super().__init__(**data)

    def matches_restaurant(self, restaurant: Restaurant) -> bool:
        """Check if restaurant matches these criteria"""

        # Distance check
        if restaurant.distance_from(self.location) > self.radius_km:
            return False

        # Category check
        if self.categories:
            if not any(restaurant.matches_category(cat) for cat in self.categories):
                return False

        # Rating check
        if restaurant.rating < self.min_rating:
            return False

        # Price level check
        if self.price_levels and restaurant.price_level:
            if restaurant.price_level not in self.price_levels:
                return False

        # Opening hours check
        if self.must_be_open and not restaurant.is_open_now:
            return False

        # Feature requirements
        for feature in self.required_features:
            if not getattr(restaurant.features, feature, False):
                return False

        return True