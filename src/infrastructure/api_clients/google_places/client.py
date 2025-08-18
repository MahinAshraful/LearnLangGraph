# src/infrastructure/api_clients/google_places/client.py
import json
import math
import random
import time
from typing import List, Optional, Dict, Any, Tuple
import logging

from ..base_client import CacheableAPIClient, APIResponse
from ....models.restaurant import Restaurant, RestaurantCategory, PriceLevel, RestaurantFeatures, PopularityData, \
    OpeningHours
from ....models.common import Location, CacheKey
from ....config.settings import get_settings
from ....infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage

logger = logging.getLogger(__name__)


class NearbySearchRequest:
    """Request object for nearby search"""

    def __init__(self, location: Tuple[float, float], radius: int, keyword: str = None,
                 min_price: Optional[int] = None, max_price: Optional[int] = None, open_now: bool = False):
        self.location = location
        self.radius = radius
        self.keyword = keyword
        self.min_price = min_price
        self.max_price = max_price
        self.open_now = open_now


class GooglePlacesClient(CacheableAPIClient):
    """Google Places API client with LLM-powered cuisine classification - FULLY DYNAMIC"""

    def __init__(self, api_key: Optional[str] = None, cache_adapter=None, use_mock: bool = False,
                 openai_client: Optional[OpenAIClient] = None):
        super().__init__(
            cache_adapter=cache_adapter,
            base_url="https://maps.googleapis.com/maps/api/place",
            api_key=api_key,
            rate_limit_per_minute=100,
            timeout_seconds=30
        )

        self.use_mock = use_mock or not api_key
        self.openai_client = openai_client  # 🤖 Add OpenAI client

        if self.use_mock:
            self.mock_restaurants = self._generate_mock_restaurants()
            logger.info("Google Places client initialized with mock data")
        else:
            logger.info("Google Places client initialized with real API")

    async def health_check(self) -> bool:
        """Check if the Google Places API is accessible"""

        if self.use_mock:
            return True

        try:
            # Simple test query
            params = {
                "location": "40.7128,-74.0060",  # NYC
                "radius": "1000",
                "type": "restaurant",
                "key": self.api_key
            }

            response = await self.get("/nearbysearch/json", params=params)
            return response.success

        except Exception as e:
            logger.error(f"Google Places health check failed: {e}")
            return False

    async def nearby_search(self, request: NearbySearchRequest) -> APIResponse[List[Restaurant]]:
        """Search for nearby restaurants"""

        if self.use_mock:
            return await self._mock_nearby_search(request)
        else:
            return await self._real_nearby_search(request)

    async def _mock_nearby_search(self, request: NearbySearchRequest) -> APIResponse[List[Restaurant]]:
        """Mock nearby search with generated data"""

        start_time = time.time()

        # Filter mock restaurants based on request
        filtered_restaurants = []

        for restaurant in self.mock_restaurants:
            # Check distance
            distance = self._calculate_distance(
                request.location[0], request.location[1],
                restaurant.location.latitude, restaurant.location.longitude
            )

            if distance * 1000 > request.radius:  # Convert km to meters
                continue

            # Check keyword match
            if request.keyword:
                keyword_lower = request.keyword.lower()
                if (keyword_lower not in restaurant.name.lower() and
                        keyword_lower not in restaurant.primary_category.value.lower()):
                    continue

            # Check price range
            if request.min_price and restaurant.price_level and restaurant.price_level.value < request.min_price:
                continue
            if request.max_price and restaurant.price_level and restaurant.price_level.value > request.max_price:
                continue

            filtered_restaurants.append(restaurant)

        # Limit results
        filtered_restaurants = filtered_restaurants[:20]

        return APIResponse.success_response(
            data=filtered_restaurants,
            response_time_ms=(time.time() - start_time) * 1000
        )

    async def _real_nearby_search(self, request: NearbySearchRequest) -> APIResponse[List[Restaurant]]:
        """Real nearby search using Google Places API"""

        start_time = time.time()

        try:
            print(f"DEBUG REAL API: Starting search for keyword='{request.keyword}'")
            print(f"DEBUG REAL API: Location={request.location}, radius={request.radius}")
            print(f"DEBUG REAL API: API key present: {'Yes' if self.api_key else 'No'}")

            # Build request parameters
            params = {
                "location": f"{request.location[0]},{request.location[1]}",
                "radius": str(request.radius),
                "type": "restaurant",
                "key": self.api_key
            }

            # Add keyword if specified
            if request.keyword:
                params["keyword"] = request.keyword
                print(f"DEBUG REAL API: Adding keyword parameter: {request.keyword}")

            # Add price range if specified
            if request.min_price:
                params["minprice"] = str(request.min_price)
            if request.max_price:
                params["maxprice"] = str(request.max_price)

            # Add open now if specified
            if request.open_now:
                params["opennow"] = "true"

            print(f"DEBUG REAL API: Final params: {params}")

            # Make API call
            print(f"DEBUG REAL API: Making request to /nearbysearch/json")
            response = await self.get("/nearbysearch/json", params=params)

            print(f"DEBUG REAL API: Response success: {response.success}")
            print(f"DEBUG REAL API: Response data type: {type(response.data)}")

            if response.success and isinstance(response.data, dict):
                results = response.data.get("results", [])
                print(f"DEBUG REAL API: Number of results from Google: {len(results)}")
                if results:
                    print(f"DEBUG REAL API: First result: {results[0].get('name', 'No name')}")
                else:
                    print(f"DEBUG REAL API: Unexpected response data: {response.data}")

            if not response.success:
                print(f"DEBUG REAL API: API call failed: {response.error}")
                return response

            # Convert Google Places response to our Restaurant model
            restaurants = []
            raw_results = response.data.get("results", []) if isinstance(response.data, dict) else []

            print(f"DEBUG REAL API: Processing {len(raw_results)} places from Google")

            for i, place in enumerate(raw_results):
                try:
                    print(f"DEBUG REAL API: Converting place {i}: {place.get('name', 'No name')}")
                    restaurant = await self._convert_place_to_restaurant(place)  # 🤖 Now async!
                    restaurants.append(restaurant)
                    print(f"DEBUG REAL API: Successfully converted: {restaurant.name}")
                except Exception as e:
                    print(f"DEBUG REAL API: Failed to convert place {i}: {e}")
                    logger.warning(f"Failed to convert place to restaurant: {e}")
                    continue

            print(f"DEBUG REAL API: Final restaurant count: {len(restaurants)}")

            return APIResponse.success_response(
                data=restaurants,
                response_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            print(f"DEBUG REAL API: Exception occurred: {e}")
            logger.error(f"Real Google Places API call failed: {e}")
            return APIResponse.error_response(f"Google Places API error: {str(e)}")

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
            restaurant = await self._convert_place_to_restaurant(place_data, detailed=True)
            return APIResponse.success_response(
                data=restaurant,
                response_time_ms=response.response_time_ms
            )
        except Exception as e:
            return APIResponse.error_response(f"Failed to parse place details: {e}")


    async def _classify_restaurant_cuisine(self, restaurant_name: str,
                                           place_types: List[str]) -> RestaurantCategory:
        """Use LLM to classify restaurant cuisine - COMPLETELY DYNAMIC with smart pattern recognition"""

        print(f"DEBUG LLM: Classifying cuisine for: {restaurant_name}")

        # Create cache key
        cache_key = f"cuisine_classification:{restaurant_name.lower().replace(' ', '_')}"

        # 🔧 FIX: Use correct cache adapter attribute name
        cache_adapter = getattr(self, 'cache_adapter', None) or getattr(self, '_cache_adapter', None)

        # Check cache first
        if cache_adapter:
            try:
                cached_result = await cache_adapter.get(cache_key)
                if cached_result:
                    print(f"DEBUG LLM: Cache hit for {restaurant_name} -> {cached_result}")
                    try:
                        return RestaurantCategory(cached_result)
                    except ValueError:
                        pass  # Cache had invalid value, continue to LLM
            except Exception as e:
                print(f"DEBUG LLM: Cache access failed: {e}")

        # If no OpenAI client, fallback
        if not self.openai_client:
            print(f"DEBUG LLM: No OpenAI client, using fallback for {restaurant_name}")
            return self._fallback_cuisine_detection(restaurant_name, place_types)

        # 🧠 SMART PROMPT - TEACHES LLM TO RECOGNIZE PATTERNS (NO HARD-CODED LISTS!)
        prompt = f"""You are an expert at identifying restaurant cuisine types from names. Analyze this restaurant:

Name: "{restaurant_name}"
Google types: {place_types}

Use these pattern recognition clues to determine cuisine:

Italian clues: Words like Mario, Luigi, Nonna, Casa, Da, Del, Della, Ristorante, Osteria, Trattoria, Palazzo, names ending in -o/-a/-i
Mexican clues: Words like Taco, Casa, El/La/Los/Las, Cantina, Hacienda, Mexican place names
Japanese clues: Sushi, Ramen, Yakitori, Sake, Japanese names, Zen, Fuji, Tokyo, Osaka
Chinese clues: Golden, Dragon, Palace, Dynasty, Garden, House, Panda, Lucky, Fortune
French clues: Bistro, Cafe, Chez, Maison, French names, Le/La
Thai clues: Thai, Pad, Spice, Bangkok, Siam, Lemongrass
Indian clues: Curry, Tandoor, Masala, Palace, Indian place names
Mediterranean clues: Olive, Gyro, Kebab, Mediterranean, Greek names
Korean clues: Korean, BBQ, Seoul, Bulgogi, Kimchi
Vietnamese clues: Pho, Saigon, Vietnamese names

Think step by step:
1. What language/culture does the name suggest?
2. Are there specific food words in the name?
3. What cuisine pattern is most likely?

Respond with exactly one word: italian, mexican, japanese, chinese, thai, indian, american, french, mediterranean, korean, vietnamese, pizza, sushi, seafood, steakhouse, cafe, bar, bakery, or american."""

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.openai_client.chat_completion(
                messages,
                max_tokens=15,
                temperature=0.2  # Slightly higher for reasoning
            )

            if response.success:
                cuisine_str = response.data.content.strip().lower()
                print(f"DEBUG LLM: {restaurant_name} -> LLM reasoned '{cuisine_str}'")

                # 🤖 DYNAMIC MAPPING - NO HARD-CODED DICTIONARY!
                for category in RestaurantCategory:
                    if category.value.lower() == cuisine_str:
                        print(f"DEBUG LLM: Matched to category: {category.value}")

                        # Cache the result
                        if cache_adapter:
                            try:
                                await cache_adapter.set(cache_key, category.value, ttl=86400)
                            except Exception as e:
                                print(f"DEBUG LLM: Cache set failed: {e}")

                        return category

                # If no exact match, try partial matching
                for category in RestaurantCategory:
                    if cuisine_str in category.value.lower() or category.value.lower() in cuisine_str:
                        print(f"DEBUG LLM: Partial match to category: {category.value}")
                        if cache_adapter:
                            try:
                                await cache_adapter.set(cache_key, category.value, ttl=86400)
                            except Exception as e:
                                print(f"DEBUG LLM: Cache set failed: {e}")
                        return category

                # If still no match, fallback
                print(f"DEBUG LLM: No match for '{cuisine_str}', using fallback")
                return self._fallback_cuisine_detection(restaurant_name, place_types)
            else:
                print(f"DEBUG LLM: LLM call failed for {restaurant_name}: {response.error}")
                return self._fallback_cuisine_detection(restaurant_name, place_types)

        except Exception as e:
            print(f"DEBUG LLM: Exception classifying {restaurant_name}: {e}")
            return self._fallback_cuisine_detection(restaurant_name, place_types)

    def _fallback_cuisine_detection(self, restaurant_name: str, place_types: List[str]) -> RestaurantCategory:
        """Simple fallback if LLM is unavailable - MINIMAL HARD-CODING FOR SAFETY"""

        name_lower = restaurant_name.lower()

        # Only the most obvious cases for safety
        if any(word in name_lower for word in ['pizza', 'pizzeria']):
            return RestaurantCategory.PIZZA
        elif any(word in name_lower for word in ['sushi', 'ramen']):
            return RestaurantCategory.JAPANESE
        elif any(word in name_lower for word in ['taco', 'burrito']):
            return RestaurantCategory.MEXICAN
        elif any(word in name_lower for word in ['cafe', 'coffee']):
            return RestaurantCategory.CAFE
        elif any(word in name_lower for word in ['bar', 'pub', 'tavern']):
            return RestaurantCategory.BAR
        else:
            return RestaurantCategory.AMERICAN


    async def _convert_place_to_restaurant(self, place_data: dict, detailed: bool = False) -> Restaurant:
        """Convert Google place with LLM-based cuisine detection"""

        print(f"DEBUG CONVERT: Converting place: {place_data.get('name', 'No name')}")
        print(f"DEBUG CONVERT: Place types: {place_data.get('types', [])}")

        # Extract basic information
        place_id = place_data.get("place_id", "")
        name = place_data.get("name", "Unknown Restaurant")

        # Extract location from geometry
        geometry = place_data.get("geometry", {})
        location_data = geometry.get("location", {})
        location = Location(
            latitude=location_data.get("lat", 0.0),
            longitude=location_data.get("lng", 0.0),
            address=place_data.get("vicinity", ""),
            city="New York",  # Default for now
            state="NY",
            country="US"
        )

        # Map price_level (0-4 in Google) to our PriceLevel enum (1-4)
        google_price_level = place_data.get("price_level")
        if google_price_level is not None and google_price_level > 0:
            price_level = PriceLevel(google_price_level)
        else:
            price_level = None

        # 🤖 USE LLM FOR CUISINE CLASSIFICATION - COMPLETELY DYNAMIC!
        types = place_data.get("types", [])
        category = await self._classify_restaurant_cuisine(name, types)

        print(f"DEBUG CONVERT: Detected category: {category.value}")

        # Extract other fields
        rating = place_data.get("rating", 0.0)
        user_ratings_total = place_data.get("user_ratings_total", 0)
        formatted_address = place_data.get("formatted_address", place_data.get("vicinity", ""))

        # Create restaurant features
        features = RestaurantFeatures(
            outdoor_seating=False,  # Not available in basic search
            wifi=False,
            accepts_reservations=price_level and price_level.value >= 2,
            delivery_available=False,
            takeout_available=True,
            good_for_groups=False,
            wheelchair_accessible=False
        )

        # Create popularity data
        popularity = PopularityData(
            current_popularity=None,
            is_usually_busy_now=None
        )

        # Extract photos
        photos = []
        if "photos" in place_data:
            for photo in place_data["photos"][:3]:  # Limit to 3 photos
                if "photo_reference" in photo:
                    photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo['photo_reference']}&key={self.api_key}"
                    photos.append(photo_url)

        restaurant = Restaurant(
            place_id=place_id,
            name=name,
            location=location,
            primary_category=category,
            price_level=price_level,
            rating=rating,
            user_ratings_total=user_ratings_total,
            phone_number=None,  # Not in basic search
            website=None,  # Not in basic search
            formatted_address=formatted_address,
            opening_hours=None,  # Not in basic search
            features=features,
            popularity=popularity,
            photos=photos,
            data_source="google_places"
        )

        print(f"DEBUG CONVERT: Created restaurant: {restaurant.name} ({restaurant.primary_category.value})")
        return restaurant

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""

        # Haversine formula
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Radius of earth in kilometers
        r = 6371
        return c * r

    # ============================================================================
    # FOR TESTING PURPOSES ONLY - Mock Restaurant Generation
    # ============================================================================

    def _generate_mock_restaurants(self) -> List[Restaurant]:
        """Generate mock restaurant data - FOR TESTING PURPOSES ONLY"""

        restaurants = []

        # NYC coordinates as base
        base_lat, base_lng = 40.7128, -74.0060

        # FOR TESTING PURPOSES ONLY - Generate one restaurant per cuisine type
        for i, category in enumerate(RestaurantCategory):
            if i >= 20:  # Limit to 20 restaurants for testing
                break

            # FOR TESTING PURPOSES ONLY - Generate generic name based on cuisine type
            cuisine_name = category.value.replace('_', ' ').title()
            restaurant_name = f"Test {cuisine_name} Restaurant {i + 1}"

            # FOR TESTING PURPOSES ONLY - Random location variation
            lat_offset = random.uniform(-0.1, 0.1)  # ~11km
            lng_offset = random.uniform(-0.1, 0.1)

            location = Location(
                latitude=base_lat + lat_offset,
                longitude=base_lng + lng_offset,
                address=f"{random.randint(1, 999)} Test Street {i + 1}",
                city="New York",
                state="NY",
                country="US"
            )

            # FOR TESTING PURPOSES ONLY - Random realistic data
            price_level = PriceLevel(random.randint(1, 4))
            rating = round(random.uniform(3.5, 4.8), 1)

            # FOR TESTING PURPOSES ONLY - Create restaurant features
            features = RestaurantFeatures(
                outdoor_seating=random.choice([True, False]),
                wifi=random.choice([True, False]),
                accepts_reservations=price_level.value >= 3,
                delivery_available=random.choice([True, False]),
                takeout_available=True,
                good_for_groups=random.choice([True, False]),
                wheelchair_accessible=random.choice([True, False])
            )

            # FOR TESTING PURPOSES ONLY - Create popularity data
            popularity = PopularityData(
                current_popularity=random.randint(20, 100),
                is_usually_busy_now=random.choice([True, False])
            )

            restaurant = Restaurant(
                place_id=f"test_mock_{i}",
                name=restaurant_name,
                location=location,
                primary_category=category,
                price_level=price_level,
                rating=rating,
                user_ratings_total=random.randint(50, 2000),
                phone_number=f"+1-555-TEST-{random.randint(1000, 9999)}",
                website=f"https://www.test-restaurant-{i}.com",
                formatted_address=f"{location.address}, {location.city}, {location.state}",
                opening_hours=None,
                features=features,
                popularity=popularity,
                photos=[f"https://test.example.com/photo_{i}_{j}.jpg" for j in range(3)],
                data_source="test_mock"
            )

            restaurants.append(restaurant)

        return restaurants

    # ============================================================================
    # Other utility methods (unchanged)
    # ============================================================================

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

            # Convert to our day name (0=Sunday -> "sunday")
            day_index = (day + 6) % 7  # Convert Sunday=0 to Sunday=6
            if day_index >= len(day_names):
                continue

            day_name = day_names[day_index]

            # Extract time
            open_hour = open_time.get("time", "0000")
            close_hour = close_time.get("time", "2359") if close_time else "2359"

            # Set hours (this is simplified - real implementation would be more robust)
            setattr(opening_hours, f"{day_name}_open", open_hour)
            setattr(opening_hours, f"{day_name}_close", close_hour)

        return opening_hours