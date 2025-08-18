import json
from typing import Dict, Any, List, Optional
import logging

from .base_node import BaseNode
from ..recommendation_state import RecommendationState
from ...infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage
from ...models.query import ParsedQuery, QueryType, LocationPreference, TimePreference, SocialContext
from ...models.restaurant import RestaurantCategory, PriceLevel
from ...models.user import DietaryRestriction, AmbiancePreference
from ...models.common import Urgency, DistancePreference

import traceback

def debug_log_error(e, location):
    print(f"DEBUG: Error in {location}: {e}")
    print(f"DEBUG: Error type: {type(e)}")
    print(f"DEBUG: Traceback:")
    traceback.print_exc()

logger = logging.getLogger(__name__)


class QueryParserNode(BaseNode):
    """100% LLM-based query parser - NO hard-coded keywords"""

    def __init__(self, openai_client: OpenAIClient):
        super().__init__("query_parser")
        self.openai_client = openai_client

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Parse user query using pure LLM understanding"""

        user_query = state.get("user_query", "")
        if not user_query:
            return self._handle_error(state, "No user query provided", is_fatal=True)

        try:
            print(f"DEBUG: Starting LLM-based parsing for: '{user_query}'")

            # Pure LLM parsing - no rule-based fallback
            parsed_query = await self._llm_only_parsing(user_query)

            print(f"DEBUG: LLM parsing completed with confidence: {parsed_query.confidence}")

            # Calculate complexity and reasoning needs
            complexity_score = self._calculate_complexity_score(parsed_query)
            should_use_smart_reasoning = complexity_score > 0.7 or parsed_query.confidence < 0.6

            logger.info(f"Parsed query: {parsed_query.query_type} (confidence: {parsed_query.confidence:.2f})")

            return {
                "parsed_query": parsed_query,
                "complexity_score": complexity_score,
                "should_use_smart_reasoning": should_use_smart_reasoning,
                **self._update_performance_tracking(state, tokens_used=150)  # Estimate
            }

        except Exception as e:
            logger.error(f"LLM query parsing failed: {e}")
            return self._handle_error(state, f"Failed to parse query: {str(e)}", is_fatal=True)

    async def _llm_only_parsing(self, user_query: str) -> ParsedQuery:
        """Pure LLM parsing with comprehensive understanding"""

        system_prompt = """You are an expert restaurant query parser. Analyze the user's request and extract ALL relevant information.

Think step-by-step:
1. What is their PRIMARY intent?
2. What specific preferences did they mention?
3. What context clues can you detect?
4. What's most important to them?

Extract this information and return ONLY valid JSON:

{
  "query_type": "cuisine_specific|location_based|occasion_based|feature_based|price_based|time_based|mood_based|social_based|dietary_based|experience_based|general",
  "confidence": 0.0-1.0,

  "cuisines": ["italian", "mexican", "chinese", "japanese", "thai", "indian", "american", "french", "mediterranean", "korean", "vietnamese", "pizza", "sushi", "seafood", "steakhouse", "cafe", "bar", "bakery"],

  "price_preferences": ["budget", "moderate", "expensive", "fine_dining"],

  "dietary_restrictions": ["vegetarian", "vegan", "gluten_free", "dairy_free", "nut_free", "halal", "kosher", "keto", "paleo"],

  "ambiance": ["romantic", "casual", "family_friendly", "business", "trendy", "quiet", "lively", "cozy", "outdoor"],

  "required_features": ["outdoor_seating", "live_music", "parking", "delivery", "reservations", "wifi", "wheelchair_accessible"],

  "location_info": {
    "max_distance_km": 10.0,
    "distance_preference": "walking|nearby|city_wide|no_preference",
    "specific_areas": ["neighborhood1", "neighborhood2"]
  },

  "time_context": {
    "urgency": "now|soon|today|this_week|planning",
    "meal_type": "breakfast|brunch|lunch|dinner|late_night",
    "flexible": true
  },

  "social_context": {
    "party_size": 2,
    "occasion": "date|business|family|celebration|casual",
    "companion_types": ["family", "friends", "business", "romantic"]
  },

  "quality_preferences": {
    "min_rating": 0.0,
    "prefer_popular": false,
    "prefer_hidden_gems": false
  }
}

Use null for any information not mentioned. Be precise and confident."""

        user_message = f"Parse this restaurant query: '{user_query}'"

        response = await self.openai_client.chat_completion([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message)
        ], max_tokens=600, temperature=0.1)

        if not response.success:
            # Fallback to basic parsing
            return self._create_fallback_parsed_query(user_query)

        try:
            llm_data = json.loads(response.data.content)
            return self._convert_llm_data_to_parsed_query(user_query, llm_data)

        except json.JSONDecodeError:
            print(f"DEBUG: JSON decode failed, using fallback for: {user_query}")
            return self._create_fallback_parsed_query(user_query)

    def _convert_llm_data_to_parsed_query(self, user_query: str, llm_data: Dict) -> ParsedQuery:
        """Convert LLM JSON response to ParsedQuery object"""

        # Map query type
        query_type = self._map_to_query_type(llm_data.get("query_type", "general"))

        # Map cuisines to enums
        cuisines = []
        for cuisine in llm_data.get("cuisines", []):
            if cuisine:
                mapped_cuisine = self._map_to_cuisine_enum(cuisine)
                if mapped_cuisine:
                    cuisines.append(mapped_cuisine)

        # Map price preferences
        price_levels = []
        for price in llm_data.get("price_preferences", []):
            if price:
                mapped_price = self._map_to_price_enum(price)
                if mapped_price:
                    price_levels.append(mapped_price)

        # Map dietary restrictions
        dietary_restrictions = []
        for dietary in llm_data.get("dietary_restrictions", []):
            if dietary:
                mapped_dietary = self._map_to_dietary_enum(dietary)
                if mapped_dietary:
                    dietary_restrictions.append(mapped_dietary)

        # Map ambiance preferences
        ambiance_prefs = []
        for ambiance in llm_data.get("ambiance", []):
            if ambiance:
                mapped_ambiance = self._map_to_ambiance_enum(ambiance)
                if mapped_ambiance:
                    ambiance_prefs.append(mapped_ambiance)

        # Extract location preference
        location_info = llm_data.get("location_info", {})
        location_pref = LocationPreference(
            max_distance_km=location_info.get("max_distance_km", 10.0),
            distance_preference=self._map_to_distance_preference(
                location_info.get("distance_preference", "nearby")
            )
        )

        # Extract time preference
        time_info = llm_data.get("time_context", {})
        time_pref = TimePreference(
            urgency=self._map_to_urgency(time_info.get("urgency", "planning")),
            meal_type=time_info.get("meal_type"),
            flexible_timing=time_info.get("flexible", True)
        )

        # Extract social context
        social_info = llm_data.get("social_context", {})
        social_context = SocialContext(
            party_size=social_info.get("party_size", 2),
            occasion=social_info.get("occasion"),
            companion_types=social_info.get("companion_types", [])
        )

        # Extract quality preferences
        quality_info = llm_data.get("quality_preferences", {})
        min_rating = quality_info.get("min_rating", 0.0)
        prefer_popular = quality_info.get("prefer_popular", False)
        prefer_hidden_gems = quality_info.get("prefer_hidden_gems", False)

        return ParsedQuery(
            original_query=user_query,
            query_type=query_type,
            confidence=llm_data.get("confidence", 0.7),
            cuisine_preferences=cuisines,
            price_preferences=price_levels,
            dietary_requirements=dietary_restrictions,
            ambiance_preferences=ambiance_prefs,
            location_preference=location_pref,
            time_preference=time_pref,
            social_context=social_context,
            required_features=llm_data.get("required_features", []),
            min_rating=min_rating,
            prefer_popular=prefer_popular,
            prefer_hidden_gems=prefer_hidden_gems,
            max_results=10
        )

    def _create_fallback_parsed_query(self, user_query: str) -> ParsedQuery:
        """Create basic fallback when LLM fails"""

        return ParsedQuery(
            original_query=user_query,
            query_type=QueryType.GENERAL,
            confidence=0.5,
            cuisine_preferences=[],
            price_preferences=[],
            dietary_requirements=[],
            ambiance_preferences=[],
            location_preference=LocationPreference(),
            time_preference=TimePreference(),
            social_context=SocialContext(),
            required_features=[],
            min_rating=0.0,
            max_results=10
        )

    # Mapping helper methods
    def _map_to_query_type(self, query_type_str: str) -> QueryType:
        """Map string to QueryType enum"""
        mapping = {
            "cuisine_specific": QueryType.CUISINE_SPECIFIC,
            "location_based": QueryType.LOCATION_BASED,
            "occasion_based": QueryType.OCCASION_BASED,
            "feature_based": QueryType.FEATURE_BASED,
            "price_based": QueryType.PRICE_BASED,
            "time_based": QueryType.TIME_BASED,
            "mood_based": QueryType.MOOD_BASED,
            "social_based": QueryType.SOCIAL_BASED,
            "dietary_based": QueryType.DIETARY_BASED,
            "experience_based": QueryType.EXPERIENCE_BASED,
            "general": QueryType.GENERAL
        }
        return mapping.get(query_type_str, QueryType.GENERAL)

    def _map_to_cuisine_enum(self, cuisine_str: str) -> Optional[RestaurantCategory]:
        """Map string to RestaurantCategory enum"""
        cuisine_mapping = {
            "italian": RestaurantCategory.ITALIAN,
            "mexican": RestaurantCategory.MEXICAN,
            "chinese": RestaurantCategory.CHINESE,
            "japanese": RestaurantCategory.JAPANESE,
            "thai": RestaurantCategory.THAI,
            "indian": RestaurantCategory.INDIAN,
            "american": RestaurantCategory.AMERICAN,
            "french": RestaurantCategory.FRENCH,
            "mediterranean": RestaurantCategory.MEDITERRANEAN,
            "korean": RestaurantCategory.KOREAN,
            "vietnamese": RestaurantCategory.VIETNAMESE,
            "pizza": RestaurantCategory.PIZZA,
            "sushi": RestaurantCategory.SUSHI,
            "seafood": RestaurantCategory.SEAFOOD,
            "steakhouse": RestaurantCategory.STEAKHOUSE,
            "cafe": RestaurantCategory.CAFE,
            "bar": RestaurantCategory.BAR,
            "bakery": RestaurantCategory.BAKERY
        }
        return cuisine_mapping.get(cuisine_str.lower())

    def _map_to_price_enum(self, price_str: str) -> Optional[PriceLevel]:
        """Map string to PriceLevel enum"""
        price_mapping = {
            "budget": PriceLevel.INEXPENSIVE,
            "moderate": PriceLevel.MODERATE,
            "expensive": PriceLevel.EXPENSIVE,
            "fine_dining": PriceLevel.VERY_EXPENSIVE
        }
        return price_mapping.get(price_str.lower())

    def _map_to_dietary_enum(self, dietary_str: str) -> Optional[DietaryRestriction]:
        """Map string to DietaryRestriction enum"""
        dietary_mapping = {
            "vegetarian": DietaryRestriction.VEGETARIAN,
            "vegan": DietaryRestriction.VEGAN,
            "gluten_free": DietaryRestriction.GLUTEN_FREE,
            "dairy_free": DietaryRestriction.DAIRY_FREE,
            "nut_free": DietaryRestriction.NUT_FREE,
            "halal": DietaryRestriction.HALAL,
            "kosher": DietaryRestriction.KOSHER,
            "keto": DietaryRestriction.KETO,
            "paleo": DietaryRestriction.PALEO
        }
        return dietary_mapping.get(dietary_str.lower())

    def _map_to_ambiance_enum(self, ambiance_str: str) -> Optional[AmbiancePreference]:
        """Map string to AmbiancePreference enum"""
        ambiance_mapping = {
            "romantic": AmbiancePreference.ROMANTIC,
            "casual": AmbiancePreference.CASUAL,
            "family_friendly": AmbiancePreference.FAMILY_FRIENDLY,
            "business": AmbiancePreference.BUSINESS,
            "trendy": AmbiancePreference.TRENDY,
            "quiet": AmbiancePreference.QUIET,
            "lively": AmbiancePreference.LIVELY,
            "cozy": AmbiancePreference.COZY,
            "outdoor": AmbiancePreference.OUTDOOR
        }
        return ambiance_mapping.get(ambiance_str.lower())

    def _map_to_distance_preference(self, distance_str: str) -> DistancePreference:
        """Map string to DistancePreference enum"""
        distance_mapping = {
            "walking": DistancePreference.WALKING,
            "nearby": DistancePreference.NEARBY,
            "city_wide": DistancePreference.CITY_WIDE,
            "no_preference": DistancePreference.NO_PREFERENCE
        }
        return distance_mapping.get(distance_str.lower(), DistancePreference.NEARBY)

    def _map_to_urgency(self, urgency_str: str) -> Urgency:
        """Map string to Urgency enum"""
        urgency_mapping = {
            "now": Urgency.NOW,
            "soon": Urgency.SOON,
            "today": Urgency.TODAY,
            "this_week": Urgency.THIS_WEEK,
            "planning": Urgency.PLANNING
        }
        return urgency_mapping.get(urgency_str.lower(), Urgency.PLANNING)

    def _calculate_complexity_score(self, parsed_query: ParsedQuery) -> float:
        """Calculate query complexity for routing decisions"""

        complexity = 0.0

        # Multiple cuisines increase complexity
        complexity += len(parsed_query.cuisine_preferences) * 0.1

        # Dietary restrictions add complexity
        complexity += len(parsed_query.dietary_requirements) * 0.15

        # Multiple ambiance preferences
        complexity += len(parsed_query.ambiance_preferences) * 0.1

        # Required features add complexity
        complexity += len(parsed_query.required_features) * 0.1

        # Low confidence increases complexity
        if parsed_query.confidence < 0.7:
            complexity += 0.3

        # Urgent queries are more complex
        if parsed_query.time_preference.urgency in [Urgency.NOW, Urgency.SOON]:
            complexity += 0.2

        return min(complexity, 1.0)
