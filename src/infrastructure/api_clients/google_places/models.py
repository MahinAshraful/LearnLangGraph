# src/infrastructure/api_clients/google_places/models.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ....models.common import BaseModel as DomainBaseModel


@dataclass
class GooglePlacesSearchRequest:
    """Request model for Google Places search"""
    location: str  # "lat,lng"
    radius: int = 5000
    type: str = "restaurant"
    keyword: Optional[str] = None
    language: str = "en"
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    open_now: bool = False
    page_token: Optional[str] = None


class PlaceGeometry(DomainBaseModel):
    """Google Places geometry information"""
    location: Dict[str, float] = Field(..., description="Lat/lng coordinates")
    viewport: Optional[Dict[str, Any]] = Field(None, description="Viewport bounds")


class PlacePhoto(DomainBaseModel):
    """Google Places photo reference"""
    photo_reference: str = Field(..., description="Photo reference string")
    height: int = Field(..., description="Photo height")
    width: int = Field(..., description="Photo width")
    html_attributions: List[str] = Field(default_factory=list)


class PlaceOpeningHours(DomainBaseModel):
    """Google Places opening hours"""
    open_now: Optional[bool] = Field(None, description="Currently open")
    periods: List[Dict[str, Any]] = Field(default_factory=list, description="Opening periods")
    weekday_text: List[str] = Field(default_factory=list, description="Human readable hours")


class PlaceReview(DomainBaseModel):
    """Google Places review"""
    author_name: str = Field(..., description="Reviewer name")
    author_url: Optional[str] = Field(None, description="Reviewer profile URL")
    language: str = Field(..., description="Review language")
    profile_photo_url: Optional[str] = Field(None, description="Reviewer photo URL")
    rating: int = Field(..., ge=1, le=5, description="Review rating")
    relative_time_description: str = Field(..., description="Time description")
    text: str = Field(..., description="Review text")
    time: int = Field(..., description="Review timestamp")


class Place(DomainBaseModel):
    """Google Places basic place information"""

    # Core identifiers
    place_id: str = Field(..., description="Google Places ID")
    name: str = Field(..., description="Place name")

    # Location
    geometry: PlaceGeometry = Field(..., description="Place geometry")
    formatted_address: Optional[str] = Field(None, description="Formatted address")
    vicinity: Optional[str] = Field(None, description="Vicinity description")

    # Ratings and popularity
    rating: Optional[float] = Field(None, ge=1, le=5, description="Average rating")
    user_ratings_total: Optional[int] = Field(None, ge=0, description="Total ratings")

    # Pricing
    price_level: Optional[int] = Field(None, ge=0, le=4, description="Price level")

    # Classification
    types: List[str] = Field(default_factory=list, description="Place types")

    # Media
    photos: List[PlacePhoto] = Field(default_factory=list, description="Place photos")

    # Contact
    formatted_phone_number: Optional[str] = Field(None, description="Phone number")
    international_phone_number: Optional[str] = Field(None, description="International phone")
    website: Optional[str] = Field(None, description="Website URL")

    # Operational
    opening_hours: Optional[PlaceOpeningHours] = Field(None, description="Opening hours")
    permanently_closed: Optional[bool] = Field(None, description="Permanently closed")
    business_status: Optional[str] = Field(None, description="Business status")

    # Additional details (from Place Details API)
    reviews: List[PlaceReview] = Field(default_factory=list, description="Place reviews")
    url: Optional[str] = Field(None, description="Google Maps URL")
    utc_offset: Optional[int] = Field(None, description="UTC offset in minutes")

    @property
    def coordinates(self) -> tuple[float, float]:
        """Get coordinates as (lat, lng) tuple"""
        location = self.geometry.location
        return (location["lat"], location["lng"])

    @property
    def is_restaurant(self) -> bool:
        """Check if this place is a restaurant"""
        restaurant_types = {
            "restaurant", "food", "establishment",
            "meal_takeaway", "meal_delivery"
        }
        return any(ptype in restaurant_types for ptype in self.types)

    @property
    def has_reviews(self) -> bool:
        """Check if place has reviews"""
        return self.user_ratings_total and self.user_ratings_total > 0


class PlaceDetails(Place):
    """Extended place information from Place Details API"""

    # Additional contact info
    adr_address: Optional[str] = Field(None, description="Structured address")

    # Additional operational info
    current_opening_hours: Optional[PlaceOpeningHours] = Field(None, description="Current opening hours")
    secondary_opening_hours: List[PlaceOpeningHours] = Field(default_factory=list, description="Secondary hours")

    # Accessibility and amenities
    wheelchair_accessible_entrance: Optional[bool] = Field(None, description="Wheelchair accessible")

    # Delivery and takeout
    delivery: Optional[bool] = Field(None, description="Offers delivery")
    dine_in: Optional[bool] = Field(None, description="Offers dine-in")
    takeout: Optional[bool] = Field(None, description="Offers takeout")

    # Reservations and payments
    reservable: Optional[bool] = Field(None, description="Accepts reservations")
    serves_beer: Optional[bool] = Field(None, description="Serves beer")
    serves_wine: Optional[bool] = Field(None, description="Serves wine")
    serves_lunch: Optional[bool] = Field(None, description="Serves lunch")
    serves_dinner: Optional[bool] = Field(None, description="Serves dinner")
    serves_breakfast: Optional[bool] = Field(None, description="Serves breakfast")
    serves_brunch: Optional[bool] = Field(None, description="Serves brunch")

    # Editorial summary
    editorial_summary: Optional[Dict[str, str]] = Field(None, description="Editorial summary")


class GooglePlacesSearchResponse(DomainBaseModel):
    """Response from Google Places search APIs"""

    results: List[Place] = Field(default_factory=list, description="Search results")
    status: str = Field(..., description="API status")

    # Pagination
    next_page_token: Optional[str] = Field(None, description="Token for next page")

    # Error information
    error_message: Optional[str] = Field(None, description="Error message if status != OK")
    info_messages: List[str] = Field(default_factory=list, description="Info messages")

    # Request metadata
    html_attributions: List[str] = Field(default_factory=list, description="Required attributions")

    @property
    def is_successful(self) -> bool:
        """Check if the request was successful"""
        return self.status == "OK"

    @property
    def has_results(self) -> bool:
        """Check if response has results"""
        return len(self.results) > 0

    @property
    def restaurant_results(self) -> List[Place]:
        """Filter results to only restaurants"""
        return [place for place in self.results if place.is_restaurant]


class GooglePlacesDetailsResponse(DomainBaseModel):
    """Response from Google Places Details API"""

    result: Optional[PlaceDetails] = Field(None, description="Place details")
    status: str = Field(..., description="API status")

    # Error information  
    error_message: Optional[str] = Field(None, description="Error message if status != OK")
    info_messages: List[str] = Field(default_factory=list, description="Info messages")

    # Request metadata
    html_attributions: List[str] = Field(default_factory=list, description="Required attributions")

    @property
    def is_successful(self) -> bool:
        """Check if the request was successful"""
        return self.status == "OK"

    @property
    def has_result(self) -> bool:
        """Check if response has a result"""
        return self.result is not None


class GooglePlacesAutocompleteResponse(DomainBaseModel):
    """Response from Google Places Autocomplete API"""

    predictions: List[Dict[str, Any]] = Field(default_factory=list, description="Autocomplete predictions")
    status: str = Field(..., description="API status")

    # Error information
    error_message: Optional[str] = Field(None, description="Error message if status != OK")
    info_messages: List[str] = Field(default_factory=list, description="Info messages")


class PopularTimesData(DomainBaseModel):
    """Popular times data structure (not officially part of Google Places API)"""

    # Day of week data (0 = Sunday, 6 = Saturday)
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week")

    # Hourly popularity data (0-100 scale, None if closed)
    hourly_data: List[Optional[int]] = Field(..., description="24-hour popularity data")

    # Additional metadata
    day_name: str = Field(..., description="Day name")
    is_today: bool = Field(default=False, description="Is this today")


class GooglePlacesError(Exception):
    """Google Places API specific errors"""

    def __init__(self, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.status = status
        self.message = message
        self.details = details or {}
        super().__init__(f"Google Places API Error [{status}]: {message}")


# Status code mappings for error handling
GOOGLE_PLACES_STATUS_CODES = {
    "OK": "The request was successful",
    "ZERO_RESULTS": "No results found",
    "OVER_QUERY_LIMIT": "You are over your quota",
    "REQUEST_DENIED": "The request was denied",
    "INVALID_REQUEST": "The request was invalid",
    "NOT_FOUND": "The referenced location was not found",
    "UNKNOWN_ERROR": "An unknown error occurred"
}


def create_google_places_error(status: str, message: Optional[str] = None) -> GooglePlacesError:
    """Create a Google Places error with appropriate message"""
    error_message = message or GOOGLE_PLACES_STATUS_CODES.get(status, "Unknown error")
    return GooglePlacesError(status, error_message)