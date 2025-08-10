from typing import List, Optional, Tuple
import logging

from ..base_client import CacheableAPIClient, APIResponse
from ....models.restaurant import Restaurant
from ....models.common import Location, CacheKey
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


class FoursquareClient(CacheableAPIClient):
    """Foursquare Places API client"""

    def __init__(self, api_key: Optional[str] = None, cache_adapter=None):
        print(f"DEBUG FOURSQUARE: Received api_key parameter: {repr(api_key)}")

        settings = get_settings()
        print(f"DEBUG FOURSQUARE: settings.api.foursquare_api_key: {repr(settings.api.foursquare_api_key)}")

        self.api_key = api_key or settings.api.foursquare_api_key
        print(f"DEBUG FOURSQUARE: Final self.api_key: {repr(self.api_key)}")

        if not self.api_key:
            raise ValueError("Foursquare API key is required")

        super().__init__(
            cache_adapter=cache_adapter,
            base_url="https://places-api.foursquare.com",
            rate_limit_per_minute=settings.api.api_rate_limits.get("foursquare", 950),
            timeout_seconds=30
        )

        logger.info("Foursquare Places client initialized")

    def _get_headers(self):
        """Override base headers for Foursquare API requirements"""

        if not self.api_key:
            logger.error("API key is None!")
            raise ValueError("Foursquare API key is required")

        # Remove fsq3 prefix if present
        api_key_without_prefix = self.api_key
        if self.api_key.startswith("fsq3"):
            api_key_without_prefix = self.api_key[4:]

        # Return Foursquare-specific headers (don't call super() to avoid conflicts)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RestaurantRecommender/1.0.0",
            "Authorization": f"Bearer {api_key_without_prefix}",
            "X-Places-Api-Version": "2025-06-17"
        }

        return headers

    async def nearby_search(self, request) -> APIResponse[List[Restaurant]]:
        """Search for nearby restaurants with proper category filtering"""

        # Convert your NearbySearchRequest to Foursquare parameters
        params = {
            "ll": f"{request.location[0]},{request.location[1]}",
            "radius": request.radius,
            "limit": 50
        }

        # Add keyword if specified - this will help filter for restaurants
        if request.keyword:
            params["query"] = f"{request.keyword}"
        else:
            # If no specific keyword, search for restaurants
            params["query"] = "restaurant"

        # Make API call
        response = await self.get("/places/search", params=params)

        if not response.success:
            return response

        # Convert Foursquare response to Restaurant objects
        restaurants = []
        for place in response.data.get("results", []):
            try:
                restaurant = self._convert_foursquare_to_restaurant(place)
                if restaurant:  # Only add if conversion successful
                    restaurants.append(restaurant)
            except Exception as e:
                logger.warning(f"Failed to convert place: {e}")
                continue

        return APIResponse.success_response(
            data=restaurants,
            response_time_ms=response.response_time_ms
        )
    def _convert_foursquare_to_restaurant(self, place_data) -> Optional[Restaurant]:
        """Convert Foursquare place to Restaurant model"""

        # Extract basic info
        place_id = place_data["fsq_place_id"]
        name = place_data["name"]

        # Create location
        location = Location(
            latitude=place_data["latitude"],
            longitude=place_data["longitude"],
            address=place_data.get("location", {}).get("address", ""),
            city=place_data.get("location", {}).get("locality", ""),
            state=place_data.get("location", {}).get("region", ""),
            country=place_data.get("location", {}).get("country", "")
        )

        # Determine primary category
        primary_category = self._map_foursquare_category(place_data.get("categories", []))

        # Skip if not a restaurant
        if not self._is_restaurant_category(place_data.get("categories", [])):
            return None

        # Create Restaurant object (basic version - we'll enhance with details later)
        restaurant = Restaurant(
            place_id=place_id,
            name=name,
            location=location,
            primary_category=primary_category,
            rating=0.0,  # Will get from details API
            user_ratings_total=0,  # Will get from details API
            phone_number=place_data.get("tel"),
            website=place_data.get("website"),
            formatted_address=place_data.get("location", {}).get("formatted_address", ""),
            data_source="foursquare"
        )

        return restaurant

    def _is_restaurant_category(self, categories) -> bool:
        """Check if this place is a restaurant with improved detection"""

        # Restaurant-related keywords (more comprehensive)
        restaurant_keywords = [
            # Direct restaurant types
            "restaurant", "cafe", "café", "bistro", "diner", "eatery",
            "dining", "grill", "kitchen", "bar", "pub", "tavern",

            # Cuisine types
            "pizza", "burger", "mexican", "italian", "chinese",
            "japanese", "thai", "indian", "sushi", "barbecue",
            "steakhouse", "seafood", "american", "french",

            # Food service types
            "fast food", "food truck", "bakery", "coffee", "sandwich",
            "noodle", "curry", "ramen", "taco", "wings"
        ]

        for category in categories:
            category_name = category.get("name", "").lower()

            # Check if any restaurant keyword is in the category name
            if any(keyword in category_name for keyword in restaurant_keywords):
                return True

            # Special cases
            if "lounge" in category_name and "hotel" not in category_name:
                return True  # Standalone lounges often serve food

        return False

    def _map_foursquare_category(self, categories):
        """Map Foursquare categories to our RestaurantCategory enum with better coverage"""
        from ....models.restaurant import RestaurantCategory

        # More comprehensive mapping
        category_mapping = {
            # Cuisine types
            "mexican": RestaurantCategory.MEXICAN,
            "italian": RestaurantCategory.ITALIAN,
            "chinese": RestaurantCategory.CHINESE,
            "japanese": RestaurantCategory.JAPANESE,
            "sushi": RestaurantCategory.SUSHI,
            "thai": RestaurantCategory.THAI,
            "indian": RestaurantCategory.INDIAN,
            "french": RestaurantCategory.FRENCH,
            "american": RestaurantCategory.AMERICAN,
            "mediterranean": RestaurantCategory.MEDITERRANEAN,
            "korean": RestaurantCategory.KOREAN,
            "vietnamese": RestaurantCategory.VIETNAMESE,

            # Food types
            "pizza": RestaurantCategory.PIZZA,
            "burger": RestaurantCategory.AMERICAN,
            "steakhouse": RestaurantCategory.STEAKHOUSE,
            "seafood": RestaurantCategory.SEAFOOD,
            "barbecue": RestaurantCategory.BBQ,
            "bbq": RestaurantCategory.BBQ,

            # Service types
            "cafe": RestaurantCategory.CAFE,
            "café": RestaurantCategory.CAFE,
            "coffee": RestaurantCategory.CAFE,
            "bar": RestaurantCategory.BAR,
            "fast food": RestaurantCategory.FAST_FOOD,
            "bakery": RestaurantCategory.BAKERY,

            # Generic
            "restaurant": RestaurantCategory.AMERICAN,
            "dining": RestaurantCategory.AMERICAN,
            "grill": RestaurantCategory.AMERICAN
        }

        for category in categories:
            category_name = category.get("name", "").lower()

            # Direct mapping
            for keyword, enum_value in category_mapping.items():
                if keyword in category_name:
                    return enum_value

            # Special patterns
            if "new american" in category_name:
                return RestaurantCategory.AMERICAN
            elif "hotel bar" in category_name:
                return RestaurantCategory.BAR

        return RestaurantCategory.AMERICAN  # Default fallback

    async def health_check(self) -> bool:
        """Check Foursquare API health"""
        try:
            response = await self.get("/places/search?ll=40.7128,-74.0060&limit=1")
            return response.success
        except Exception:
            return False