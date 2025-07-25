# src/infrastructure/api_clients/google_places/client.py

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import logging
import asyncio

from ..base_client import BaseAPIClient, APIResponse, CacheableAPIClient, with_retry
from ....config.settings import get_settings
from ....config.constants import (
    GOOGLE_PLACES_BASE_URL, GOOGLE_PLACES_TYPES, MAJOR_CITIES,
    PRICE_LEVEL_RANGES, TYPICAL_CUISINE_PRICE_LEVELS
)
from ....models.restaurant import Restaurant, RestaurantCategory, PriceLevel, Location, OpeningHours, \
    RestaurantFeatures, PopularityData
from ....models.common import TimeSlot, CacheKey
from .models import GooglePlacesSearchRequest, GooglePlacesSearchResponse, PlaceDetails

logger = logging.getLogger(__name__)


@dataclass
class NearbySearchRequest:
    """Request for nearby restaurant search"""
    location: Tuple[float, float]  # (lat, lng)
    radius: int = 5000  # meters
    restaurant_type: str = "restaurant"
    keyword: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    open_now: bool = False
    page_token: Optional[str] = None


class GooglePlacesClient(CacheableAPIClient):
    """Google Places API client with mock and real implementations"""

    def __init__(self, api_key: Optional[str] = None, cache_adapter=None, use_mock: bool = True):
        settings = get_settings()

        self.api_key = api_key or settings.api.google_places_api_key
        self.use_mock = use_mock or not self.api_key

        super().__init__(
            cache_adapter=cache_adapter,
            base_url=GOOGLE_PLACES_BASE_URL,
            api_key=self.api_key,
            rate_limit_per_minute=settings.api.api_rate_limits.get("google_places", 100),
            timeout_seconds=30
        )

        # Mock data for development
        if self.use_mock:
            self.mock_restaurants = self._generate_mock_restaurants()
            logger.info("Google Places client initialized in MOCK mode")
        else:
            logger.info("Google Places client initialized in REAL mode")

    async def health_check(self) -> bool:
        """Check Google Places API health"""
        if self.use_mock:
            return True

        try:
            # Simple test search
            request = NearbySearchRequest(
                location=(40.7128, -74.0060),  # NYC
                radius=1000
            )
            response = await self.nearby_search(request)
            return response.success
        except Exception as e:
            logger.error(f"Google Places health check failed: {e}")
            return False

    @with_retry(max_retries=2, backoff_factor=1.0)
    async def nearby_search(self, request: NearbySearchRequest) -> APIResponse[List[Restaurant]]:
        """Search for nearby restaurants"""

        if self.use_mock:
            return await self._mock_nearby_search(request)
        else:
            return await self._real_nearby_search(request)

    async def _mock_nearby_search(self, request: NearbySearchRequest) -> APIResponse[List[Restaurant]]:
        """Mock implementation for development"""

        import time
        start_time = time.time()

        # Create cache key
        cache_key = CacheKey(
            prefix="gplaces_nearby",
            identifier=f"{request.location[0]:.4f},{request.location[1]:.4f}_{request.radius}_{request.keyword or 'all'}"
        )

        # Check cache first
        if self.cache:
            cached_result = await self.cache.get(str(cache_key))
            if cached_result:
                restaurants = [Restaurant(**data) for data in cached_result]
                return APIResponse.success_response(
                    data=restaurants,
                    cached=True,
                    response_time_ms=(time.time() - start_time) * 1000
                )

        # Simulate API delay
        await asyncio.sleep(0.1)

        # Filter mock restaurants by location and criteria
        lat, lng = request.location
        filtered_restaurants = []

        for restaurant in self.mock_restaurants:
            # Calculate distance (simplified)
            distance_km = self._calculate_distance(
                lat, lng,
                restaurant.location.latitude,
                restaurant.location.longitude
            )

            # Apply filters
            if distance_km * 1000 > request.radius:
                continue

            if request.keyword:
                keyword_lower = request.keyword.lower()
                if (keyword_lower not in restaurant.name.lower() and
                        keyword_lower not in restaurant.primary_category.value.lower()):
                    continue

            if request.min_price and restaurant.price_level:
                if restaurant.price_level.value < request.min_price:
                    continue

            if request.max_price and restaurant.price_level:
                if restaurant.price_level.value > request.max_price:
                    continue

            if request.open_now and not restaurant.is_open_now:
                continue

            filtered_restaurants.append(restaurant)

        # Sort by distance and rating
        filtered_restaurants.sort(
            key=lambda r: (
                self._calculate_distance(lat, lng, r.location.latitude, r.location.longitude),
                -r.rating
            )
        )

        # Limit results
        results = filtered_restaurants[:20]

        # Cache results
        if self.cache:
            cache_data = [restaurant.dict() for restaurant in results]
            await self.cache.set(
                str(cache_key),
                cache_data,
                ttl=self.settings.cache.ttl.get("google_places", 900)
            )

        response_time_ms = (time.time() - start_time) * 1000

        return APIResponse.success_response(
            data=results,
            response_time_ms=response_time_ms
        )

    async def _real_nearby_search(self, request: NearbySearchRequest) -> APIResponse[List[Restaurant]]:
        """Real Google Places API implementation"""

        # Prepare API parameters
        params = {
            "location": f"{request.location[0]},{request.location[1]}",
            "radius": request.radius,
            "type": request.restaurant_type,
            "key": self.api_key
        }

        if request.keyword:
            params["keyword"] = request.keyword

        if request.min_price:
            params["minprice"] = request.min_price

        if request.max_price:
            params["maxprice"] = request.max_price

        if request.open_now:
            params["opennow"] = "true"

        if request.page_token:
            params["pagetoken"] = request.page_token

        # Make API call
        response = await self.get("/nearbysearch/json", params=params)

        if not response.success:
            return response

        # Convert Google Places response to our Restaurant model
        restaurants = []
        for place in response.data.get("results", []):
            try:
                restaurant = self._convert_place_to_restaurant(place)
                restaurants.append(restaurant)
            except Exception as e:
                logger.warning(f"Failed to convert place to restaurant: {e}")
                continue

        return APIResponse.success_response(
            data=restaurants,
            response_time_ms=response.response_time_ms
        )

    async def get_place_details(self, place_id: str, fields: Optional[List[str]] = None) -> APIResponse[Restaurant]:
        """Get detailed information about a specific place"""

        if self.use_mock:
            return await self._mock_place_details(place_id)
        else:
            return await self._real_place_details(place_id, fields)

    async def _mock_place_details(self, place_id: str) -> APIResponse[Restaurant]:
        """Mock place details"""

        # Find restaurant by place_id
        for restaurant in self.mock_restaurants:
            if restaurant.place_id == place_id:
                return APIResponse.success_response(data=restaurant)

        return APIResponse.error_response(f"Place not found: {place_id}")

    async def _real_place_details(self, place_id: str, fields: Optional[List[str]] = None) -> APIResponse[Restaurant]:
        """Real place details from Google Places API"""

        if not fields:
            fields = [
                "place_id", "name", "rating", "user_ratings_total",
                "price_level", "formatted_address", "geometry",
                "formatted_phone_number", "website", "opening_hours",
                "photos", "types", "reviews"
            ]

        params = {
            "place_id": place_id,
            "fields": ",".join(fields),
            "key": self.api_key
        }

        response = await self.get("/details/json", params=params)

        if not response.success:
            return response

        place_data = response.data.get("result")
        if not place_data:
            return APIResponse.error_response("Place not found")

        try:
            restaurant = self._convert_place_to_restaurant(place_data, detailed=True)
            return APIResponse.success_response(
                data=restaurant,
                response_time_ms=response.response_time_ms
            )
        except Exception as e:
            return APIResponse.error_response(f"Failed to parse place details: {e}")

    def _convert_place_to_restaurant(self, place_data: Dict[str, Any], detailed: bool = False) -> Restaurant:
        """Convert Google Places API response to Restaurant model"""

        # Extract basic info
        place_id = place_data["place_id"]
        name = place_data["name"]
        rating = place_data.get("rating", 0.0)
        user_ratings_total = place_data.get("user_ratings_total", 0)
        price_level = place_data.get("price_level")

        # Extract location
        geometry = place_data.get("geometry", {})
        location_data = geometry.get("location", {})
        location = Location(
            latitude=location_data.get("lat", 0.0),
            longitude=location_data.get("lng", 0.0),
            address=place_data.get("formatted_address", ""),
            city=self._extract_city_from_address(place_data.get("formatted_address", ""))
        )

        # Determine primary category
        types = place_data.get("types", [])
        primary_category = self._map_google_types_to_category(types)

        # Create restaurant features
        features = RestaurantFeatures()
        if detailed:
            # Extract features from detailed response
            features = self._extract_features_from_place(place_data)

        # Extract opening hours
        opening_hours = None
        if "opening_hours" in place_data:
            opening_hours = self._convert_opening_hours(place_data["opening_hours"])

        # Extract photos
        photos = []
        if "photos" in place_data:
            photos = [
                f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo['photo_reference']}&key={self.api_key}"
                for photo in place_data["photos"][:5]  # Limit to 5 photos
            ]

        # Create popularity data
        popularity = PopularityData(
            current_popularity=random.randint(20, 90),  # Mock data
            is_usually_busy_now=random.choice([True, False, None])
        )

        return Restaurant(
            place_id=place_id,
            name=name,
            location=location,
            primary_category=primary_category,
            google_types=types,
            price_level=PriceLevel(price_level) if price_level else None,
            rating=rating,
            user_ratings_total=user_ratings_total,
            phone_number=place_data.get("formatted_phone_number"),
            website=place_data.get("website"),
            formatted_address=place_data.get("formatted_address", ""),
            opening_hours=opening_hours,
            features=features,
            popularity=popularity,
            photos=photos,
            data_source="google_places"
        )

    def _map_google_types_to_category(self, types: List[str]) -> RestaurantCategory:
        """Map Google Places types to our restaurant categories"""

        # Priority mapping for cuisine types
        cuisine_mapping = {
            "italian_restaurant": RestaurantCategory.ITALIAN,
            "chinese_restaurant": RestaurantCategory.CHINESE,
            "japanese_restaurant": RestaurantCategory.JAPANESE,
            "mexican_restaurant": RestaurantCategory.MEXICAN,
            "thai_restaurant": RestaurantCategory.THAI,
            "indian_restaurant": RestaurantCategory.INDIAN,
            "french_restaurant": RestaurantCategory.FRENCH,
            "mediterranean_restaurant": RestaurantCategory.MEDITERRANEAN,
            "korean_restaurant": RestaurantCategory.KOREAN,
            "vietnamese_restaurant": RestaurantCategory.VIETNAMESE,
            "pizza": RestaurantCategory.PIZZA,
            "sushi_restaurant": RestaurantCategory.SUSHI,
            "steakhouse": RestaurantCategory.STEAKHOUSE,
            "seafood_restaurant": RestaurantCategory.SEAFOOD,
            "vegetarian_restaurant": RestaurantCategory.VEGETARIAN,
            "fast_food_restaurant": RestaurantCategory.FAST_FOOD,
            "cafe": RestaurantCategory.CAFE,
            "bar": RestaurantCategory.BAR,
            "bakery": RestaurantCategory.BAKERY
        }

        # Check for specific cuisine types first
        for gtype in types:
            if gtype in cuisine_mapping:
                return cuisine_mapping[gtype]

        # Default mappings
        if "restaurant" in types:
            return RestaurantCategory.AMERICAN  # Default
        elif "meal_takeaway" in types:
            return RestaurantCategory.FAST_FOOD
        elif "cafe" in types:
            return RestaurantCategory.CAFE
        elif "bar" in types:
            return RestaurantCategory.BAR
        elif "bakery" in types:
            return RestaurantCategory.BAKERY
        else:
            return RestaurantCategory.AMERICAN  # Default fallback

    def _extract_features_from_place(self, place_data: Dict[str, Any]) -> RestaurantFeatures:
        """Extract restaurant features from place data"""

        features = RestaurantFeatures()

        # This would be more sophisticated in a real implementation
        # For now, randomly assign some features based on place type and rating
        types = place_data.get("types", [])
        rating = place_data.get("rating", 0)

        # Higher rated places more likely to have features
        feature_probability = min(rating / 5.0, 1.0)

        if random.random() < feature_probability:
            features.wifi = True

        if "bar" in types or rating > 4.0:
            features.has_bar = random.choice([True, False])

        if rating > 4.2:
            features.accepts_reservations = True

        # Randomly assign other features
        features.outdoor_seating = random.choice([True, False])
        features.delivery_available = random.choice([True, False])
        features.takeout_available = True
        features.wheelchair_accessible = random.choice([True, False])
        features.good_for_groups = random.choice([True, False])
        features.good_for_kids = "family" in str(place_data).lower()

        return features

    def _convert_opening_hours(self, opening_hours_data: Dict[str, Any]) -> Optional[OpeningHours]:
        """Convert Google Places opening hours to our format"""

        periods = opening_hours_data.get("periods", [])
        if not periods:
            return None

        # Initialize opening hours
        opening_hours = OpeningHours()
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        for period in periods:
            if "open" not in period:
                continue

            open_time = period["open"]
            close_time = period.get("close")

            day = open_time.get("day")  # 0=Sunday, 1=Monday, etc.
            if day is None:
                continue

            # Convert Google day (0=Sunday) to our format (0=Monday)
            our_day = (day + 6) % 7
            day_name = day_names[our_day]

            # Parse times
            open_hour_min = open_time.get("time", "0000")
            open_hour = int(open_hour_min[:2])
            open_min = int(open_hour_min[2:])

            if close_time:
                close_hour_min = close_time.get("time", "2359")
                close_hour = int(close_hour_min[:2])
                close_min = int(close_hour_min[2:])
            else:
                # Open 24 hours
                close_hour, close_min = 23, 59

            time_slot = TimeSlot(
                start_time=time(open_hour, open_min),
                end_time=time(close_hour, close_min),
                day_of_week=our_day
            )

            # Add to appropriate day
            current_slots = getattr(opening_hours, day_name) or []
            current_slots.append(time_slot)
            setattr(opening_hours, day_name, current_slots)

        return opening_hours

    def _extract_city_from_address(self, address: str) -> Optional[str]:
        """Extract city from formatted address"""
        if not address:
            return None

        # Simple extraction - look for common patterns
        parts = address.split(", ")
        if len(parts) >= 2:
            # Usually city is second to last part
            return parts[-2].split()[0]  # Remove state/zip

        return None

    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in kilometers"""
        import math

        # Haversine formula
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        # Radius of earth in kilometers
        r = 6371
        return c * r

    def _generate_mock_restaurants(self) -> List[Restaurant]:
        """Generate mock restaurant data for development"""

        restaurants = []

        # NYC coordinates as base
        base_lat, base_lng = 40.7128, -74.0060

        # Restaurant templates
        restaurant_templates = [
            # Italian
            {"name": "Mario's Authentic Italian", "category": RestaurantCategory.ITALIAN, "price": 3, "rating": 4.4},
            {"name": "Little Italy Bistro", "category": RestaurantCategory.ITALIAN, "price": 2, "rating": 4.1},
            {"name": "Nonna's Kitchen", "category": RestaurantCategory.ITALIAN, "price": 2, "rating": 4.3},
            {"name": "Palazzo Ristorante", "category": RestaurantCategory.ITALIAN, "price": 4, "rating": 4.6},

            # Asian
            {"name": "Golden Dragon", "category": RestaurantCategory.CHINESE, "price": 2, "rating": 4.0},
            {"name": "Sakura Sushi", "category": RestaurantCategory.JAPANESE, "price": 3, "rating": 4.5},
            {"name": "Ramen House", "category": RestaurantCategory.JAPANESE, "price": 2, "rating": 4.2},
            {"name": "Thai Spice", "category": RestaurantCategory.THAI, "price": 2, "rating": 4.3},
            {"name": "Pho Saigon", "category": RestaurantCategory.VIETNAMESE, "price": 1, "rating": 4.0},
            {"name": "Curry Palace", "category": RestaurantCategory.INDIAN, "price": 2, "rating": 4.1},

            # American
            {"name": "The Burger Joint", "category": RestaurantCategory.AMERICAN, "price": 2, "rating": 4.2},
            {"name": "Steakhouse 21", "category": RestaurantCategory.STEAKHOUSE, "price": 4, "rating": 4.3},
            {"name": "City Diner", "category": RestaurantCategory.AMERICAN, "price": 2, "rating": 3.9},
            {"name": "BBQ Pit", "category": RestaurantCategory.BBQ, "price": 2, "rating": 4.1},

            # Mexican
            {"name": "El Mariachi", "category": RestaurantCategory.MEXICAN, "price": 2, "rating": 4.0},
            {"name": "Taco Bell Express", "category": RestaurantCategory.MEXICAN, "price": 1, "rating": 3.7},
            {"name": "Azteca Grill", "category": RestaurantCategory.MEXICAN, "price": 3, "rating": 4.2},

            # Others
            {"name": "Café Parisien", "category": RestaurantCategory.FRENCH, "price": 3, "rating": 4.4},
            {"name": "Mediterranean Breeze", "category": RestaurantCategory.MEDITERRANEAN, "price": 2, "rating": 4.2},
            {"name": "Corner Coffee Shop", "category": RestaurantCategory.CAFE, "price": 1, "rating": 3.8},
        ]

        for i, template in enumerate(restaurant_templates):
            # Generate realistic coordinates around NYC
            lat_offset = random.uniform(-0.05, 0.05)  # ~5km radius
            lng_offset = random.uniform(-0.05, 0.05)

            location = Location(
                latitude=base_lat + lat_offset,
                longitude=base_lng + lng_offset,
                address=f"{100 + i} {random.choice(['Main St', 'Broadway', 'Park Ave', 'Madison Ave'])}, New York, NY 10001",
                city="New York",
                state="NY",
                country="US"
            )

            # Generate opening hours
            opening_hours = self._generate_mock_opening_hours()

            # Generate features
            features = RestaurantFeatures(
                outdoor_seating=random.choice([True, False]),
                wifi=random.choice([True, False]),
                accepts_reservations=template["price"] >= 3,
                delivery_available=random.choice([True, False]),
                takeout_available=True,
                good_for_groups=random.choice([True, False]),
                wheelchair_accessible=random.choice([True, False])
            )

            # Generate popularity data
            popularity = PopularityData(
                current_popularity=random.randint(20, 90),
                typical_wait_time=random.randint(5, 30) if random.choice([True, False]) else None,
                is_usually_busy_now=random.choice([True, False, None])
            )

            restaurant = Restaurant(
                place_id=f"mock_place_{i}",
                name=template["name"],
                location=location,
                primary_category=template["category"],
                price_level=PriceLevel(template["price"]),
                rating=template["rating"],
                user_ratings_total=random.randint(50, 1000),
                phone_number=f"+1-555-{1000 + i:04d}",
                website=f'https://{template["name"].lower().replace(" ", "").replace(chr(39), "")}.com',
                formatted_address=location.address,
                opening_hours=opening_hours,
                features=features,
                popularity=popularity,
                photos=[f"https://example.com/photo_{i}_{j}.jpg" for j in range(3)],
                data_source="mock"
            )

            restaurants.append(restaurant)

        return restaurants

    def _generate_mock_opening_hours(self) -> OpeningHours:
        """Generate realistic opening hours"""

        # Common restaurant hours
        weekday_open = time(11, 0)  # 11 AM
        weekday_close = time(22, 0)  # 10 PM
        weekend_open = time(10, 0)  # 10 AM
        weekend_close = time(23, 0)  # 11 PM

        weekday_slot = TimeSlot(start_time=weekday_open, end_time=weekday_close)
        weekend_slot = TimeSlot(start_time=weekend_open, end_time=weekend_close)

        return OpeningHours(
            monday=[weekday_slot],
            tuesday=[weekday_slot],
            wednesday=[weekday_slot],
            thursday=[weekday_slot],
            friday=[weekday_slot],
            saturday=[weekend_slot],
            sunday=[weekend_slot]
        )