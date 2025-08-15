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

        # Get the API key first
        resolved_api_key = api_key or settings.api.foursquare_api_key
        print(f"DEBUG FOURSQUARE: Resolved api_key: {repr(resolved_api_key)}")

        if not resolved_api_key:
            raise ValueError("Foursquare API key is required")

        # IMPORTANT: Pass api_key to parent constructor so it doesn't get overwritten
        super().__init__(
            cache_adapter=cache_adapter,
            base_url="https://places-api.foursquare.com",
            api_key=resolved_api_key,  # ← This was missing!
            rate_limit_per_minute=settings.api.api_rate_limits.get("foursquare", 950),
            timeout_seconds=30
        )

        logger.info("Foursquare Places client initialized")

    def _get_headers(self):
        """Override base headers for Foursquare API requirements"""

        if not self.api_key:
            logger.error("API key is None!")
            raise ValueError("Foursquare API key is required")

        # Remove fsq3 prefix if present (Foursquare uses this prefix for identification)
        api_key_without_prefix = self.api_key
        if self.api_key.startswith("fsq3"):
            api_key_without_prefix = self.api_key[4:]  # Remove "fsq3" prefix
            logger.debug(f"Removed fsq3 prefix from API key")

        # Return Foursquare-specific headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RestaurantRecommender/1.0.0",
            "Authorization": f"Bearer {api_key_without_prefix}",
            "X-Places-Api-Version": "2025-06-17"
        }

        return headers

    async def nearby_search(self, request) -> APIResponse[List[Restaurant]]:
        """Search for nearby restaurants and enhance top results with details"""

        # Convert your NearbySearchRequest to Foursquare parameters
        params = {
            "ll": f"{request.location[0]},{request.location[1]}",
            "radius": request.radius,
            "limit": 50
        }

        # Add keyword if specified
        if request.keyword:
            params["query"] = f"{request.keyword}"
        else:
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
                if restaurant:
                    restaurants.append(restaurant)
            except Exception as e:
                logger.warning(f"Failed to convert place: {e}")

        # Enhance top 20 restaurants with detailed data (avoid hitting API limits)
        enhanced_restaurants = []

        for i, restaurant in enumerate(restaurants[:20]):  # Only enhance top 20
            try:
                print(f"DEBUG FOURSQUARE: Enhancing {restaurant.name} with details")
                details_response = await self.get_place_details(restaurant.place_id)

                if details_response.success:
                    enhanced_restaurants.append(details_response.data)
                    print(f"DEBUG FOURSQUARE: Enhanced {restaurant.name} - rating: {details_response.data.rating}")
                else:
                    enhanced_restaurants.append(restaurant)  # Keep original if details fail

            except Exception as e:
                logger.warning(f"Failed to enhance restaurant {restaurant.name}: {e}")
                enhanced_restaurants.append(restaurant)  # Keep original

        # Add remaining restaurants without enhancement
        enhanced_restaurants.extend(restaurants[20:])

        return APIResponse.success_response(
            data=enhanced_restaurants,
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

    async def get_place_details(self, place_id: str) -> APIResponse[Restaurant]:
        """Get detailed place information including rating, price, reviews"""

        # Make API call to place details endpoint
        response = await self.get(f"/places/{place_id}")

        if not response.success:
            return response

        place_data = response.data

        try:
            restaurant = self._convert_detailed_foursquare_to_restaurant(place_data)
            return APIResponse.success_response(
                data=restaurant,
                response_time_ms=response.response_time_ms
            )
        except Exception as e:
            return APIResponse.error_response(f"Failed to parse place details: {e}")

    def _convert_detailed_foursquare_to_restaurant(self, place_data) -> Restaurant:
        """Convert detailed Foursquare place to Restaurant with ratings/price"""

        # Extract basic info
        place_id = place_data["fsq_place_id"]
        name = place_data["name"]

        # Extract rating (0-10 scale, convert to 0-5)
        rating = 0.0
        if "rating" in place_data:
            rating = min(place_data["rating"] / 2.0, 5.0)  # Convert 10-scale to 5-scale

        # Extract review count
        user_ratings_total = place_data.get("stats", {}).get("total_ratings", 0)

        # Extract price level (1-4 scale)
        price_level = None
        if "price" in place_data:
            price_value = place_data["price"]
            if price_value >= 1 and price_value <= 4:
                from ....models.restaurant import PriceLevel
                price_mapping = {1: PriceLevel.INEXPENSIVE, 2: PriceLevel.MODERATE,
                                 3: PriceLevel.EXPENSIVE, 4: PriceLevel.VERY_EXPENSIVE}
                price_level = price_mapping.get(price_value)

        # Create location
        location = Location(
            latitude=place_data["geocodes"]["main"]["latitude"],
            longitude=place_data["geocodes"]["main"]["longitude"],
            address=place_data.get("location", {}).get("address", ""),
            city=place_data.get("location", {}).get("locality", ""),
            state=place_data.get("location", {}).get("region", ""),
            country=place_data.get("location", {}).get("country", "")
        )

        # Determine category
        primary_category = self._map_foursquare_category(place_data.get("categories", []))

        # Create Restaurant with real data
        restaurant = Restaurant(
            place_id=place_id,
            name=name,
            location=location,
            primary_category=primary_category,
            rating=rating,
            user_ratings_total=user_ratings_total,
            price_level=price_level,
            phone_number=place_data.get("tel"),
            website=place_data.get("website"),
            formatted_address=place_data.get("location", {}).get("formatted_address", ""),
            data_source="foursquare"
        )

        return restaurant