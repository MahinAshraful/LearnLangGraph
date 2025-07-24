from typing import Dict, Any, List
import logging

from .base_node import BaseNode
from ..state.recommendation_state import RecommendationState
from ...domain.models.restaurant import Restaurant
from ...domain.models.query import ParsedQuery
from ...domain.models.user import UserPreferences

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
            preference_filtered = self._apply_preference_filter(quality_filtered, parsed_query, user_preferences)
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

        filtered = []

        for restaurant in restaurants:
            # Minimum rating threshold
            if restaurant.rating < 3.0:
                continue

            # Minimum review count for reliability
            if restaurant.user_ratings_total < 5:
                continue

            # Skip permanently closed restaurants
            if hasattr(restaurant, 'permanently_closed') and restaurant.permanently_closed:
                continue

            filtered.append(restaurant)

        logger.debug(f"Quality filter: {len(restaurants)} → {len(filtered)}")
        return filtered

    def _apply_preference_filter(self,
                                 restaurants: List[Restaurant],
                                 parsed_query: ParsedQuery,
                                 user_preferences: Optional[UserPreferences]) -> List[Restaurant]:
        """Apply user and query preference filters"""

        filtered = []

        for restaurant in restaurants:
            # Skip restaurants user has explicitly disliked
            if (user_preferences and
                    restaurant.place_id in user_preferences.disliked_restaurants):
                continue

            # Apply dietary restriction filters
            if parsed_query and parsed_query.dietary_requirements:
                if not self._meets_dietary_requirements(restaurant, parsed_query.dietary_requirements):
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

    def _meets_dietary_requirements(self, restaurant: Restaurant, dietary_requirements: List[str]) -> bool:
        """Check if restaurant meets dietary requirements"""

        # This would be more sophisticated in production
        # For now, simple heuristics based on cuisine and features

        for requirement in dietary_requirements:
            if requirement == "vegetarian":
                if restaurant.primary_category.value == "vegetarian":
                    continue
                # Most cuisines have vegetarian options
                if restaurant.primary_category.value in ["indian", "thai", "mediterranean", "italian"]:
                    continue
                return False

            elif requirement == "vegan":
                if restaurant.primary_category.value in ["vegetarian", "vegan"]:
                    continue
                # Limited vegan options in some cuisines
                if restaurant.primary_category.value not in ["indian", "thai", "mediterranean"]:
                    return False

            elif requirement == "gluten_free":
                # Assume most restaurants can accommodate, except very limited ones
                if restaurant.primary_category.value in ["pizza", "bakery"]:
                    return False

        return True

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