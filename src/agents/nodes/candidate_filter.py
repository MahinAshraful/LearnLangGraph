from typing import Dict, Any, List, Optional
import logging

from .base_node import BaseNode
from ..recommendation_state import RecommendationState
from ...models.restaurant import Restaurant
from ...models.query import ParsedQuery
from ...models.user import UserPreferences
from ...infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage

logger = logging.getLogger(__name__)


class CandidateFilterNode(BaseNode):
    """Filters and prepares candidate restaurants for scoring"""

    def __init__(self):
        super().__init__("candidate_filter")

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Filter nearby restaurants to create final candidate set"""

        nearby_restaurants = state.get("nearby_restaurants", [])
        parsed_query = state.get("parsed_query")
        user_preferences = state.get("user_preferences")

        if not nearby_restaurants:
            return self._handle_error(state, "No nearby restaurants to filter")

        try:
            # Apply filtering stages
            quality_filtered = self._apply_quality_filter(nearby_restaurants)
            preference_filtered = await self._apply_preference_filter(quality_filtered, parsed_query, user_preferences)
            final_candidates = self._apply_final_selection(preference_filtered, parsed_query)

            logger.info(f"Filtered restaurants: {len(nearby_restaurants)} → {len(final_candidates)} candidates")

            return {
                "candidate_restaurants": final_candidates
            }

        except Exception as e:
            logger.error(f"Candidate filtering failed: {e}")
            return self._handle_error(state, f"Failed to filter candidates: {str(e)}")

    def _apply_quality_filter(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        """Apply minimum quality standards"""

        print(f"DEBUG FILTER: Quality filter input: {len(restaurants)} restaurants")
        filtered = []

        for restaurant in restaurants:
            # Handle missing rating data gracefully
            if restaurant.rating > 0:  # Has real rating data
                if restaurant.rating < 3.0:  # Poor rating
                    print(f"DEBUG FILTER: Filtered out {restaurant.name} - low rating ({restaurant.rating})")
                    continue
            # If rating is 0 or missing, assume neutral (don't filter out)

            # Handle missing review count gracefully
            if restaurant.user_ratings_total > 0:  # Has real review data
                if restaurant.user_ratings_total < 5:  # Too few reviews
                    print(
                        f"DEBUG FILTER: Filtered out {restaurant.name} - few reviews ({restaurant.user_ratings_total})")
                    continue
            # If review count is 0 or missing, assume neutral (don't filter out)

            # Skip permanently closed restaurants
            if hasattr(restaurant, 'permanently_closed') and restaurant.permanently_closed:
                print(f"DEBUG FILTER: Filtered out {restaurant.name} - permanently closed")
                continue

            filtered.append(restaurant)

        print(f"DEBUG FILTER: Quality filter output: {len(filtered)} restaurants")
        logger.debug(f"Quality filter: {len(restaurants)} → {len(filtered)}")
        return filtered

    async def _apply_preference_filter(self,
                                     restaurants: List[Restaurant],
                                     parsed_query: ParsedQuery,
                                     user_preferences: Optional[UserPreferences]) -> List[Restaurant]:
        """Apply user and query preference filters"""

        filtered = []

        for restaurant in restaurants:
            # Skip restaurants user has explicitly disliked
            if (user_preferences and
                    hasattr(user_preferences, 'disliked_restaurants') and
                    restaurant.place_id in user_preferences.disliked_restaurants):
                continue

            # Apply dietary restriction filters
            if parsed_query and parsed_query.dietary_requirements:
                if not await self._meets_dietary_requirements(restaurant, parsed_query.dietary_requirements):
                    continue

            # Apply strict feature requirements (must-haves)
            if parsed_query and parsed_query.deal_breakers:
                if self._has_deal_breakers(restaurant, parsed_query.deal_breakers):
                    continue

            filtered.append(restaurant)

        logger.debug(f"Preference filter: {len(restaurants)} → {len(filtered)}")
        return filtered

    def _apply_final_selection(self,
                               restaurants: List[Restaurant],
                               parsed_query: ParsedQuery) -> List[Restaurant]:
        """Apply final selection and limit candidates"""

        # Sort by basic quality metrics first
        restaurants.sort(key=lambda r: (r.rating, r.user_ratings_total), reverse=True)

        # Ensure diversity in cuisine types
        diverse_candidates = self._ensure_cuisine_diversity(restaurants)

        # Limit to reasonable number for scoring (performance optimization)
        max_candidates = parsed_query.max_results * 3 if parsed_query else 30
        final_candidates = diverse_candidates[:max_candidates]

        logger.debug(f"Final selection: {len(restaurants)} → {len(final_candidates)}")
        return final_candidates

    async def _meets_dietary_requirements(self, restaurant: Restaurant, dietary_requirements: List[str]) -> bool:
        """LLM-based intelligent dietary compatibility assessment"""

        if not dietary_requirements:
            return True

        # Use LLM to intelligently assess dietary compatibility
        return await self._llm_dietary_assessment(restaurant, dietary_requirements)

    async def _llm_dietary_assessment(self, restaurant: Restaurant, dietary_requirements: List[str]) -> bool:
        """Use LLM to assess if restaurant can accommodate dietary needs"""

        # Create cache key for performance
        cache_key = f"dietary_{restaurant.place_id}_{'+'.join(sorted(dietary_requirements))}"

        # Check cache first (dietary info doesn't change often)
        if hasattr(self, '_dietary_cache') and cache_key in self._dietary_cache:
            return self._dietary_cache[cache_key]

        if not hasattr(self, '_dietary_cache'):
            self._dietary_cache = {}

        # Prepare LLM prompt with restaurant info
        restaurant_info = f"""
Restaurant: {restaurant.name}
Cuisine: {restaurant.primary_category.value}
Rating: {restaurant.rating}/5.0
Price Level: {restaurant.price_level.value if restaurant.price_level else 'Unknown'}
Google Types: {getattr(restaurant, 'google_types', [])}
"""

        dietary_list = ", ".join(dietary_requirements)

        prompt = f"""You are an expert on restaurant dietary accommodations. Assess if this restaurant can likely accommodate these dietary needs.

{restaurant_info}

Dietary Requirements: {dietary_list}

Consider:
1. Restaurant name clues (e.g., "Green Leaf Cafe" likely vegetarian-friendly)
2. Cuisine type compatibility (e.g., Indian restaurants typically have many vegetarian options)
3. Modern restaurant trends (most restaurants now accommodate common dietary needs)
4. Chain vs independent considerations
5. Price level (higher-end restaurants typically more accommodating)

Think step-by-step:
- What does the restaurant name suggest about their menu focus?
- How compatible is this cuisine type with the dietary requirements?
- Are there any obvious red flags (e.g., "Joe's BBQ Smokehouse" for vegan needs)?
- What's the likelihood they can accommodate these needs?

Return ONLY: "YES" if likely to accommodate, "NO" if unlikely, "MAYBE" if uncertain.

Be practical - most restaurants today can accommodate vegetarian/gluten-free, but be more careful with vegan, kosher, halal."""

        try:
            # Get OpenAI client from the scoring node or create one
            if hasattr(self, 'openai_client'):
                openai_client = self.openai_client
            else:
                # Import here to avoid circular imports
                from ...infrastructure.api_clients.openai.client import OpenAIClient, ChatMessage
                from ...infrastructure.databases.cache.memory_adapter import MemoryAdapter

                cache = MemoryAdapter()
                await cache.connect()
                openai_client = OpenAIClient(cache_adapter=cache)

            response = await openai_client.chat_completion([
                ChatMessage(role="system", content="You are an expert on restaurant dietary accommodations."),
                ChatMessage(role="user", content=prompt)
            ], max_tokens=50, temperature=0.1)

            if response.success:
                result = response.data.content.strip().upper()

                # Convert LLM response to boolean
                if result == "YES":
                    compatibility = True
                elif result == "NO":
                    compatibility = False
                else:  # "MAYBE" or unclear response
                    compatibility = True  # Default to inclusive for uncertain cases

                print(f"DEBUG DIETARY: {restaurant.name} + {dietary_requirements} = {result} -> {compatibility}")

                # Cache the result
                self._dietary_cache[cache_key] = compatibility
                return compatibility

        except Exception as e:
            print(f"DEBUG DIETARY: LLM assessment failed for {restaurant.name}: {e}")

        # Fallback to improved heuristics if LLM fails
        return self._improved_dietary_heuristics(restaurant, dietary_requirements)

    def _improved_dietary_heuristics(self, restaurant: Restaurant, dietary_requirements: List[str]) -> bool:
        """Improved fallback heuristics for dietary assessment"""

        cuisine = restaurant.primary_category.value.lower()
        name = restaurant.name.lower()

        for requirement in dietary_requirements:
            req_lower = requirement.lower()

            if req_lower == "vegetarian":
                # Vegetarian-friendly clues in name
                if any(word in name for word in ["green", "leaf", "garden", "veggie", "plant", "salad"]):
                    continue

                # Cuisine compatibility (expanded list)
                if cuisine in ["indian", "thai", "mediterranean", "italian", "vietnamese", "ethiopian", "middle_eastern"]:
                    continue

                # Modern restaurants typically accommodate
                if restaurant.rating >= 4.0:  # Higher-rated places usually more accommodating
                    continue

                # Red flags for vegetarian
                if any(word in name for word in ["bbq", "grill", "steakhouse", "burger", "wings", "smokehouse"]):
                    return False

            elif req_lower == "vegan":
                # Vegan-specific clues
                if any(word in name for word in ["vegan", "plant", "raw", "juice", "green"]):
                    continue

                # Cuisines more likely to have vegan options
                if cuisine in ["indian", "thai", "mediterranean", "middle_eastern", "ethiopian"]:
                    continue

                # Be more restrictive for vegan
                if cuisine in ["steakhouse", "bbq", "french", "italian"] or any(word in name for word in ["cheese", "dairy", "cream"]):
                    return False

            elif req_lower == "gluten_free":
                # Gluten-free friendly clues
                if "gluten" in name:
                    continue

                # Cuisines naturally lower in gluten
                if cuisine in ["mexican", "thai", "indian", "seafood"]:
                    continue

                # Red flags for gluten-free
                if cuisine in ["pizza", "bakery", "cafe"] and "gluten" not in name:
                    return False

            elif req_lower in ["halal", "kosher"]:
                # Religious dietary requirements - be more careful
                if req_lower in name or cuisine in ["middle_eastern", "mediterranean"]:
                    continue
                # Don't assume other restaurants can accommodate
                return False

        return True  # Default to allowing if no red flags

    def _has_deal_breakers(self, restaurant: Restaurant, deal_breakers: List[str]) -> bool:
        """Check if restaurant has any deal breaker features"""

        # This would check for negative requirements
        # e.g., "no_smoking", "not_too_loud", etc.
        return False  # Placeholder

    def _ensure_cuisine_diversity(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        """Ensure diversity in cuisine types among candidates"""

        if len(restaurants) <= 10:
            return restaurants

        # Group by cuisine
        cuisine_groups = {}
        for restaurant in restaurants:
            cuisine = restaurant.primary_category.value
            if cuisine not in cuisine_groups:
                cuisine_groups[cuisine] = []
            cuisine_groups[cuisine].append(restaurant)

        # Take top restaurants from each cuisine group
        diverse_selection = []

        # First, take the top restaurant from each cuisine
        for cuisine_restaurants in cuisine_groups.values():
            diverse_selection.append(cuisine_restaurants[0])

        # Then, take additional restaurants round-robin style
        remaining_slots = len(restaurants) - len(diverse_selection)
        cuisine_keys = list(cuisine_groups.keys())

        for i in range(remaining_slots):
            cuisine_key = cuisine_keys[i % len(cuisine_keys)]
            cuisine_restaurants = cuisine_groups[cuisine_key]

            # Find next restaurant from this cuisine that's not already selected
            for restaurant in cuisine_restaurants[1:]:  # Skip first (already selected)
                if restaurant not in diverse_selection:
                    diverse_selection.append(restaurant)
                    break

        return diverse_selection