import asyncio
from typing import Dict, Any, List, Optional
import logging

from .base_node import BaseNode
from ..recommendation_state import RecommendationState
from ...infrastructure.api_clients.google_places.client import GooglePlacesClient, NearbySearchRequest
from ...models.restaurant import Restaurant, RestaurantCategory
from ...models.common import Location
from ...models.query import ParsedQuery
from ...config.constants import DEFAULT_SEARCH_RADIUS_KM

logger = logging.getLogger(__name__)


class DataRetrievalNode(BaseNode):
    """Retrieves restaurant data from external APIs with intelligent query optimization"""

    def __init__(self, google_places_client: GooglePlacesClient):
        super().__init__("data_retrieval")
        self.google_places_client = google_places_client

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Retrieve nearby restaurants based on parsed query and user context"""

        parsed_query = state.get("parsed_query")
        user_location = state.get("user_location")

        if not parsed_query:
            return self._handle_error(state, "No parsed query available", is_fatal=True)

        # Use default NYC location if no user location
        if not user_location:
            user_location = Location(
                latitude=40.7128,
                longitude=-74.0060,
                city="New York",
                state="NY",
                country="US"
            )
            logger.info("Using default NYC location for restaurant search")

        try:
            # Create search requests based on query complexity
            search_requests = self._create_search_requests(parsed_query, user_location)

            # Execute searches (potentially in parallel)
            all_restaurants = []
            api_calls = 0

            if len(search_requests) == 1:
                # Single search
                restaurants = await self._execute_single_search(search_requests[0])
                all_restaurants.extend(restaurants)
                api_calls = 1
            else:
                # Multiple searches in parallel
                restaurants_lists = await self._execute_parallel_searches(search_requests)
                for restaurants in restaurants_lists:
                    all_restaurants.extend(restaurants)
                api_calls = len(search_requests)

            # Remove duplicates and apply basic filtering
            unique_restaurants = self._remove_duplicates(all_restaurants)
            filtered_restaurants = self._apply_basic_filters(unique_restaurants, parsed_query)

            logger.info(f"Retrieved {len(filtered_restaurants)} restaurants from {api_calls} API calls")

            return {
                "nearby_restaurants": filtered_restaurants,
                **self._update_performance_tracking(state, api_calls=api_calls)
            }

        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return self._handle_error(state, f"Failed to retrieve restaurant data: {str(e)}", is_fatal=True)

    def _create_search_requests(self, parsed_query: ParsedQuery, user_location: Location) -> List[NearbySearchRequest]:
        """Create optimized search requests based on query complexity"""

        location_tuple = (user_location.latitude, user_location.longitude)

        # Calculate search radius based on distance preference
        radius = self._calculate_search_radius(parsed_query)

        # Determine search strategy based on query type
        if parsed_query.cuisine_preferences and len(parsed_query.cuisine_preferences) == 1:
            # Single cuisine search - use keyword for better results
            cuisine = parsed_query.cuisine_preferences[0]
            keyword = self._cuisine_to_keyword(cuisine)

            return [NearbySearchRequest(
                location=location_tuple,
                radius=radius,
                keyword=keyword,
                min_price=self._get_min_price(parsed_query),
                max_price=self._get_max_price(parsed_query),
                open_now=parsed_query.time_preference.urgency in ["now", "soon"]
            )]

        elif parsed_query.cuisine_preferences and len(parsed_query.cuisine_preferences) > 1:
            # Multiple cuisines - create separate searches
            requests = []
            for cuisine in parsed_query.cuisine_preferences[:3]:  # Limit to 3 to avoid too many API calls
                keyword = self._cuisine_to_keyword(cuisine)

                requests.append(NearbySearchRequest(
                    location=location_tuple,
                    radius=radius,
                    keyword=keyword,
                    min_price=self._get_min_price(parsed_query),
                    max_price=self._get_max_price(parsed_query),
                    open_now=parsed_query.time_preference.urgency in ["now", "soon"]
                ))

            return requests

        else:
            # General search - no specific cuisine
            keyword = None

            # Use features as keywords if specified
            if parsed_query.required_features:
                # Use the first feature as keyword
                feature_keywords = {
                    "outdoor_seating": "outdoor seating",
                    "live_music": "live music",
                    "parking": "parking",
                    "delivery": "delivery"
                }
                first_feature = parsed_query.required_features[0]
                keyword = feature_keywords.get(first_feature)

            return [NearbySearchRequest(
                location=location_tuple,
                radius=radius,
                keyword=keyword,
                min_price=self._get_min_price(parsed_query),
                max_price=self._get_max_price(parsed_query),
                open_now=parsed_query.time_preference.urgency in ["now", "soon"]
            )]

    def _calculate_search_radius(self, parsed_query: ParsedQuery) -> int:
        """Calculate search radius in meters based on distance preference"""

        distance_pref = parsed_query.location_preference.distance_preference

        radius_mapping = {
            "walking": 1500,  # ~15 minutes walk
            "nearby": 5000,  # 5km
            "city_wide": 15000,  # 15km
            "no_preference": 10000  # 10km default
        }

        return radius_mapping.get(distance_pref.value, 5000)

    def _cuisine_to_keyword(self, cuisine: RestaurantCategory) -> str:
        """Convert cuisine category to search keyword"""

        keyword_mapping = {
            RestaurantCategory.ITALIAN: "italian restaurant",
            RestaurantCategory.CHINESE: "chinese restaurant",
            RestaurantCategory.JAPANESE: "japanese restaurant",
            RestaurantCategory.MEXICAN: "mexican restaurant",
            RestaurantCategory.THAI: "thai restaurant",
            RestaurantCategory.INDIAN: "indian restaurant",
            RestaurantCategory.FRENCH: "french restaurant",
            RestaurantCategory.AMERICAN: "american restaurant",
            RestaurantCategory.MEDITERRANEAN: "mediterranean restaurant",
            RestaurantCategory.KOREAN: "korean restaurant",
            RestaurantCategory.VIETNAMESE: "vietnamese restaurant",
            RestaurantCategory.PIZZA: "pizza",
            RestaurantCategory.SUSHI: "sushi",
            RestaurantCategory.BBQ: "barbecue",
            RestaurantCategory.SEAFOOD: "seafood restaurant",
            RestaurantCategory.STEAKHOUSE: "steakhouse",
            RestaurantCategory.VEGETARIAN: "vegetarian restaurant",
            RestaurantCategory.FAST_FOOD: "fast food",
            RestaurantCategory.CAFE: "cafe",
            RestaurantCategory.BAR: "bar restaurant"
        }

        return keyword_mapping.get(cuisine, cuisine.value)

    def _get_min_price(self, parsed_query: ParsedQuery) -> Optional[int]:
        """Get minimum price level from query"""
        if parsed_query.price_preferences:
            return min(p.value for p in parsed_query.price_preferences)
        return None

    def _get_max_price(self, parsed_query: ParsedQuery) -> Optional[int]:
        """Get maximum price level from query"""
        if parsed_query.price_preferences:
            return max(p.value for p in parsed_query.price_preferences)
        return None

    async def _execute_single_search(self, request: NearbySearchRequest) -> List[Restaurant]:
        """Execute a single search request"""

        try:
            response = await self.google_places_client.nearby_search(request)

            if response.success:
                return response.data
            else:
                logger.warning(f"Google Places search failed: {response.error}")
                return []

        except Exception as e:
            logger.error(f"Single search execution failed: {e}")
            return []

    async def _execute_parallel_searches(self, requests: List[NearbySearchRequest]) -> List[List[Restaurant]]:
        """Execute multiple search requests in parallel"""

        try:
            # Use asyncio.gather for parallel execution
            tasks = [self._execute_single_search(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and return valid results
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Parallel search failed: {result}")
                    valid_results.append([])
                else:
                    valid_results.append(result)

            return valid_results

        except Exception as e:
            logger.error(f"Parallel search execution failed: {e}")
            return [[] for _ in requests]

    def _remove_duplicates(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        """Remove duplicate restaurants based on place_id"""

        seen_place_ids = set()
        unique_restaurants = []

        for restaurant in restaurants:
            if restaurant.place_id not in seen_place_ids:
                seen_place_ids.add(restaurant.place_id)
                unique_restaurants.append(restaurant)

        logger.debug(f"Removed {len(restaurants) - len(unique_restaurants)} duplicates")
        return unique_restaurants

    def _apply_basic_filters(self, restaurants: List[Restaurant], parsed_query: ParsedQuery) -> List[Restaurant]:
        """Apply basic quality and requirement filters"""

        filtered = []

        for restaurant in restaurants:
            # Basic quality filter
            if restaurant.rating < 3.0 or restaurant.user_ratings_total < 10:
                continue

            # Cuisine filter (if Google Places didn't filter properly)
            if parsed_query.cuisine_preferences:
                cuisine_match = any(
                    restaurant.primary_category == cuisine or cuisine in restaurant.secondary_categories
                    for cuisine in parsed_query.cuisine_preferences
                )
                if not cuisine_match:
                    continue

            # Price filter (if Google Places didn't filter properly)
            if parsed_query.price_preferences and restaurant.price_level:
                if restaurant.price_level not in parsed_query.price_preferences:
                    continue

            # Opening hours filter for urgent requests
            if (parsed_query.time_preference.urgency == "now" and
                    not restaurant.is_open_now):
                continue

            # Required features filter (basic implementation)
            if parsed_query.required_features:
                feature_match = self._check_feature_requirements(restaurant, parsed_query.required_features)
                if not feature_match:
                    continue

            filtered.append(restaurant)

        logger.debug(f"Applied filters: {len(restaurants)} -> {len(filtered)} restaurants")
        return filtered

    def _check_feature_requirements(self, restaurant: Restaurant, required_features: List[str]) -> bool:
        """Check if restaurant meets feature requirements"""

        # Map feature requirements to restaurant attributes
        feature_mapping = {
            "outdoor_seating": "outdoor_seating",
            "live_music": "live_music",
            "parking": "parking_available",
            "wifi": "wifi",
            "delivery": "delivery_available",
            "takeout": "takeout_available",
            "reservations": "accepts_reservations",
            "wheelchair_accessible": "wheelchair_accessible"
        }

        for feature in required_features:
            mapped_feature = feature_mapping.get(feature)
            if mapped_feature and hasattr(restaurant.features, mapped_feature):
                if not getattr(restaurant.features, mapped_feature, False):
                    return False

        return True

    async def _enhance_restaurant_data(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        """Enhance restaurant data with additional details if needed"""

        # For high-value restaurants or when we have time, get detailed info
        enhanced_restaurants = []

        for restaurant in restaurants:
            try:
                # Only enhance top-rated restaurants to save API calls
                if restaurant.rating >= 4.2 and restaurant.user_ratings_total >= 100:
                    detailed_response = await self.google_places_client.get_place_details(restaurant.place_id)

                    if detailed_response.success:
                        enhanced_restaurants.append(detailed_response.data)
                    else:
                        enhanced_restaurants.append(restaurant)
                else:
                    enhanced_restaurants.append(restaurant)

            except Exception as e:
                logger.warning(f"Failed to enhance restaurant {restaurant.place_id}: {e}")
                enhanced_restaurants.append(restaurant)

        return enhanced_restaurants

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about search performance"""

        return {
            "node_name": self.name,
            "google_places_health": "unknown",  # Would check client health
            **self.get_performance_stats()
        }