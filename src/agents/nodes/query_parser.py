import json
from typing import Dict, Any, List
import logging

from .base_node import BaseNode
from ..recommendation_state import RecommendationState
from ...infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage
from ...models.query import ParsedQuery, QueryType, LocationPreference, TimePreference, SocialContext
from ...models.restaurant import RestaurantCategory, PriceLevel
from ...models.user import DietaryRestriction, AmbiancePreference
from ...models.common import Urgency, DistancePreference
from ...config.constants import CUISINE_KEYWORDS, PRICE_KEYWORDS, AMBIANCE_KEYWORDS, FEATURE_KEYWORDS

logger = logging.getLogger(__name__)


class QueryParserNode(BaseNode):
    """Parses natural language restaurant queries into structured format"""

    def __init__(self, openai_client: OpenAIClient):
        super().__init__("query_parser")
        self.openai_client = openai_client

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Parse user query into structured ParsedQuery object"""

        user_query = state.get("user_query", "")
        if not user_query:
            return self._handle_error(state, "No user query provided", is_fatal=True)

        try:
            # First try rule-based parsing for simple queries
            parsed_query = await self._try_rule_based_parsing(user_query)

            # If rule-based parsing is insufficient, use LLM
            if parsed_query.confidence < 0.7 or parsed_query.query_type == QueryType.GENERAL:
                parsed_query = await self._llm_enhanced_parsing(user_query, parsed_query)

            # Calculate complexity score
            complexity_score = parsed_query.complexity_score

            # Determine if smart reasoning is needed
            should_use_smart_reasoning = parsed_query.requires_smart_reasoning

            logger.info(f"Parsed query: {parsed_query.query_type.value} (confidence: {parsed_query.confidence:.2f})")

            return {
                "parsed_query": parsed_query,
                "complexity_score": complexity_score,
                "should_use_smart_reasoning": should_use_smart_reasoning,
                **self._update_performance_tracking(state, tokens_used=0)  # Will update after LLM call
            }

        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return self._handle_error(state, f"Failed to parse query: {str(e)}", is_fatal=True)

    async def _try_rule_based_parsing(self, query: str) -> ParsedQuery:
        """Attempt to parse query using rule-based approach"""

        query_lower = query.lower()

        # Initialize parsed query
        parsed_query = ParsedQuery(
            original_query=query,
            query_type=QueryType.GENERAL,
            confidence=0.5
        )

        # Extract cuisines
        cuisines = []
        for cuisine, keywords in CUISINE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                try:
                    cuisines.append(RestaurantCategory(cuisine))
                except ValueError:
                    continue

        if cuisines:
            parsed_query.cuisine_preferences = cuisines
            parsed_query.query_type = QueryType.CUISINE_SPECIFIC
            parsed_query.confidence = 0.8

        # Extract price preferences
        price_level = None
        for price, keywords in PRICE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                try:
                    price_level = PriceLevel[price.upper()]
                    break
                except (KeyError, ValueError):
                    continue

        if price_level:
            parsed_query.price_preferences = [price_level]
            if parsed_query.query_type == QueryType.GENERAL:
                parsed_query.query_type = QueryType.PRICE_BASED
                parsed_query.confidence = 0.7

        # Extract ambiance preferences
        ambiance_prefs = []
        for ambiance, keywords in AMBIANCE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                try:
                    ambiance_prefs.append(AmbiancePreference(ambiance))
                except ValueError:
                    continue

        if ambiance_prefs:
            parsed_query.ambiance_preferences = ambiance_prefs
            if parsed_query.query_type == QueryType.GENERAL:
                parsed_query.query_type = QueryType.MOOD_BASED
                parsed_query.confidence = 0.6

        # Extract features
        features = []
        for feature, keywords in FEATURE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                features.append(feature)

        if features:
            parsed_query.required_features = features
            if parsed_query.query_type == QueryType.GENERAL:
                parsed_query.query_type = QueryType.FEATURE_BASED
                parsed_query.confidence = 0.7

        # Extract party size
        import re
        party_size_match = re.search(r'(\d+)\s*(people|person|ppl)', query_lower)
        if party_size_match:
            party_size = int(party_size_match.group(1))
            parsed_query.social_context.party_size = min(max(party_size, 1), 20)

        # Extract urgency
        if any(word in query_lower for word in ["now", "asap", "immediately"]):
            parsed_query.time_preference.urgency = Urgency.NOW
        elif any(word in query_lower for word in ["tonight", "today"]):
            parsed_query.time_preference.urgency = Urgency.SOON

        # Location indicators
        if any(word in query_lower for word in ["near", "nearby", "close", "around"]):
            parsed_query.query_type = QueryType.LOCATION_BASED
            parsed_query.location_preference.distance_preference = DistancePreference.NEARBY

        return parsed_query

    async def _llm_enhanced_parsing(self, query: str, initial_parse: ParsedQuery) -> ParsedQuery:
        """Use LLM to enhance parsing with more sophisticated understanding"""

        system_prompt = """You are an expert at parsing restaurant search queries. Extract structured information and return valid JSON.

Analyze the user's restaurant query and extract:
1. Query type (cuisine_specific, location_based, occasion_based, feature_based, price_based, time_based, mood_based, social_based, dietary_based, experience_based, general)
2. Cuisine preferences (italian, chinese, japanese, mexican, thai, indian, french, american, mediterranean, korean, vietnamese, greek, spanish, middle_eastern, pizza, sushi, bbq, seafood, steakhouse, vegetarian, vegan, fast_food, casual_dining, fine_dining, cafe, bar, bakery)
3. Price range (budget, moderate, upscale, fine_dining)
4. Party size (number)
5. Dietary restrictions (vegetarian, vegan, gluten_free, dairy_free, nut_free, halal, kosher, keto, paleo, low_carb, low_sodium)
6. Ambiance preferences (romantic, casual, upscale, family_friendly, business, trendy, quiet, lively, cozy, outdoor, authentic, modern, traditional)
7. Required features (outdoor_seating, live_music, parking, delivery, reservations, wifi, wheelchair_accessible)
8. Urgency (now, soon, today, this_week, planning)
9. Distance preference (walking, nearby, city_wide, no_preference)
10. Special occasion or context

Return ONLY valid JSON with these exact field names. Use null for missing values."""

        user_prompt = f"""Query: "{query}"

Current rule-based parse found:
- Query type: {initial_parse.query_type.value}
- Cuisines: {[c.value for c in initial_parse.cuisine_preferences]}
- Price: {[p.value for p in initial_parse.price_preferences]}
- Features: {initial_parse.required_features}
- Party size: {initial_parse.social_context.party_size}

Please provide enhanced parsing as JSON:"""

        try:
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.settings.llm.models["query_parser"],
                temperature=self.settings.llm.temperature["query_parser"],
                max_tokens=500
            )

            if not response.success:
                logger.warning(f"LLM parsing failed: {response.error}")
                return initial_parse

            # Parse LLM response
            try:
                llm_data = json.loads(response.data.content)
                enhanced_query = self._merge_parsing_results(initial_parse, llm_data)
                enhanced_query.confidence = 0.9  # High confidence from LLM

                # Update performance tracking
                self.tokens_used = response.data.tokens_used

                return enhanced_query

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")
                return initial_parse

        except Exception as e:
            logger.error(f"LLM enhanced parsing failed: {e}")
            return initial_parse

    def _merge_parsing_results(self, initial_parse: ParsedQuery, llm_data: Dict[str, Any]) -> ParsedQuery:
        """Merge rule-based and LLM parsing results"""

        # Start with initial parse
        merged = ParsedQuery(
            original_query=initial_parse.original_query,
            query_type=initial_parse.query_type,
            cuisine_preferences=initial_parse.cuisine_preferences.copy(),
            price_preferences=initial_parse.price_preferences.copy(),
            dietary_requirements=initial_parse.dietary_requirements.copy(),
            ambiance_preferences=initial_parse.ambiance_preferences.copy(),
            required_features=initial_parse.required_features.copy(),
            location_preference=initial_parse.location_preference,
            time_preference=initial_parse.time_preference,
            social_context=initial_parse.social_context
        )

        # Update with LLM enhancements
        if llm_data.get("query_type"):
            try:
                merged.query_type = QueryType(llm_data["query_type"])
            except ValueError:
                pass

        if llm_data.get("cuisines"):
            llm_cuisines = []
            for cuisine in llm_data["cuisines"]:
                try:
                    llm_cuisines.append(RestaurantCategory(cuisine))
                except ValueError:
                    continue
            # Merge with existing cuisines
            all_cuisines = set(merged.cuisine_preferences + llm_cuisines)
            merged.cuisine_preferences = list(all_cuisines)

        if llm_data.get("price_range"):
            try:
                price_level = PriceLevel(llm_data["price_range"])
                if price_level not in merged.price_preferences:
                    merged.price_preferences.append(price_level)
            except ValueError:
                pass

        if llm_data.get("party_size") and isinstance(llm_data["party_size"], int):
            merged.social_context.party_size = min(max(llm_data["party_size"], 1), 20)

        if llm_data.get("dietary_restrictions"):
            llm_dietary = []
            for restriction in llm_data["dietary_restrictions"]:
                try:
                    llm_dietary.append(DietaryRestriction(restriction))
                except ValueError:
                    continue
            merged.dietary_requirements = list(set(merged.dietary_requirements + llm_dietary))

        if llm_data.get("ambiance_preferences"):
            llm_ambiance = []
            for ambiance in llm_data["ambiance_preferences"]:
                try:
                    llm_ambiance.append(AmbiancePreference(ambiance))
                except ValueError:
                    continue
            merged.ambiance_preferences = list(set(merged.ambiance_preferences + llm_ambiance))

        if llm_data.get("required_features"):
            merged.required_features = list(set(merged.required_features + llm_data["required_features"]))

        if llm_data.get("urgency"):
            try:
                merged.time_preference.urgency = Urgency(llm_data["urgency"])
            except ValueError:
                pass

        if llm_data.get("distance_preference"):
            try:
                merged.location_preference.distance_preference = DistancePreference(llm_data["distance_preference"])
            except ValueError:
                pass

        # Extract occasion if mentioned
        if llm_data.get("occasion"):
            merged.social_context.occasion = llm_data["occasion"]

        return merged