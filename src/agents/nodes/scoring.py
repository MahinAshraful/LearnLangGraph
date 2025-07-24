import math
from typing import Dict, Any, List, Optional
import logging

from .base_node import BaseNode
from ..state.recommendation_state import RecommendationState
from ...domain.models.restaurant import Restaurant
from ...domain.models.user import UserPreferences
from ...domain.models.query import ParsedQuery
from ...domain.models.recommendation import Recommendation, ScoreBreakdown, RecommendationReason
from ...domain.models.common import EntityId
from ...config.constants import SCORING_WEIGHTS

logger = logging.getLogger(__name__)


class ScoringNode(BaseNode):
    """Scores restaurant candidates using the 50/30/15/5 algorithm"""

    def __init__(self):
        super().__init__("scoring")

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Score all candidate restaurants using weighted algorithm"""

        candidate_restaurants = state.get("candidate_restaurants", [])
        parsed_query = state.get("parsed_query")
        user_preferences = state.get("user_preferences")
        similar_users = state.get("similar_users", [])
        collaborative_restaurants = state.get("collaborative_restaurants", [])

        if not candidate_restaurants:
            return self._handle_error(state, "No candidate restaurants to score")

        if not parsed_query:
            return self._handle_error(state, "No parsed query available for scoring")

        try:
            scored_recommendations = []

            for restaurant in candidate_restaurants:
                # Calculate all score components
                score_breakdown = await self._calculate_score_breakdown(
                    restaurant=restaurant,
                    parsed_query=parsed_query,
                    user_preferences=user_preferences,
                    similar_users=similar_users,
                    collaborative_restaurants=collaborative_restaurants
                )

                # Generate reasoning
                reasons = self._generate_recommendation_reasons(restaurant, score_breakdown, parsed_query)
                explanation = self._generate_explanation(restaurant, score_breakdown, reasons)

                # Calculate confidence based on data completeness
                confidence = self._calculate_confidence(score_breakdown, user_preferences)

                # Create recommendation
                recommendation = Recommendation(
                    id=EntityId(),
                    restaurant=restaurant,
                    score=score_breakdown,
                    primary_reasons=reasons,
                    explanation=explanation,
                    confidence=confidence,
                    rank=0,  # Will be set after sorting
                    strategy_used="hybrid"
                )

                scored_recommendations.append(recommendation)

            # Sort by total score
            scored_recommendations.sort(key=lambda r: r.score.total_score, reverse=True)

            # Assign ranks
            for i, rec in enumerate(scored_recommendations, 1):
                rec.rank = i

            logger.info(f"Scored {len(scored_recommendations)} restaurants")

            return {
                "scored_recommendations": scored_recommendations
            }

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return self._handle_error(state, f"Failed to score recommendations: {str(e)}")

    async def _calculate_score_breakdown(self,
                                         restaurant: Restaurant,
                                         parsed_query: ParsedQuery,
                                         user_preferences: Optional[UserPreferences],
                                         similar_users: List[UserPreferences],
                                         collaborative_restaurants: List[str]) -> ScoreBreakdown:
        """Calculate detailed score breakdown using 50/30/15/5 weights"""

        # Initialize score breakdown
        score_breakdown = ScoreBreakdown()

        # 1. PREFERENCE SCORE (50%)
        score_breakdown.preference_score = self._calculate_preference_score(
            restaurant, parsed_query, user_preferences, similar_users, collaborative_restaurants
        )

        # 2. CONTEXT SCORE (30%)
        score_breakdown.context_score = self._calculate_context_score(
            restaurant, parsed_query
        )

        # 3. QUALITY SCORE (15%)
        score_breakdown.quality_score = self._calculate_quality_score(restaurant)

        # 4. BOOST SCORE (5%)
        score_breakdown.boost_score = self._calculate_boost_score(
            restaurant, user_preferences, collaborative_restaurants
        )

        return score_breakdown

    def _calculate_preference_score(self,
                                    restaurant: Restaurant,
                                    parsed_query: ParsedQuery,
                                    user_preferences: Optional[UserPreferences],
                                    similar_users: List[UserPreferences],
                                    collaborative_restaurants: List[str]) -> float:
        """Calculate preference match score (50% of total)"""

        total_score = 0.0

        # Query cuisine match (40% of preference score)
        cuisine_score = 0.0
        if parsed_query.cuisine_preferences:
            for cuisine in parsed_query.cuisine_preferences:
                if restaurant.matches_category(cuisine):
                    cuisine_score = 1.0
                    break

        # User preference cuisine match (30% of preference score)
        user_cuisine_score = 0.0
        if user_preferences and user_preferences.favorite_cuisines:
            for cuisine in user_preferences.favorite_cuisines:
                if restaurant.matches_category(cuisine):
                    # Weight by user's cuisine preference strength
                    weight = user_preferences.get_cuisine_weight(cuisine)
                    user_cuisine_score = max(user_cuisine_score, weight)

        # Price preference match (20% of preference score)
        price_score = 0.0
        if parsed_query.price_preferences and restaurant.price_level:
            if restaurant.price_level in parsed_query.price_preferences:
                price_score = 1.0
        elif user_preferences and user_preferences.preferred_price_levels and restaurant.price_level:
            if restaurant.price_level in user_preferences.preferred_price_levels:
                price_score = 0.8  # Slightly lower than explicit query preference

        # Collaborative filtering (10% of preference score)
        collaborative_score = 0.0
        if restaurant.place_id in collaborative_restaurants:
            collaborative_score = 0.8
            score_breakdown.collaborative_score = collaborative_score
            score_breakdown.similar_users_count = len(similar_users)

        # Combine preference components
        total_score = (
                cuisine_score * 0.40 +
                user_cuisine_score * 0.30 +
                price_score * 0.20 +
                collaborative_score * 0.10
        )

        # Store component scores
        score_breakdown.rating_score = rating_score
        score_breakdown.popularity_score = popularity_score

        return min(total_score, 1.0)

    def _calculate_boost_score(self,
                               restaurant: Restaurant,
                               user_preferences: Optional[UserPreferences],
                               collaborative_restaurants: List[str]) -> float:
        """Calculate special boost score (5% of total)"""

        total_boost = 0.0

        # Trending/Popular boost (40% of boost score)
        trending_boost = 0.0
        if restaurant.popularity and restaurant.popularity.is_usually_busy_now:
            trending_boost = 0.6  # Popular right now
        elif restaurant.user_ratings_total > 500:
            trending_boost = 0.4  # Generally popular

        # Quality excellence boost (30% of boost score)
        excellence_boost = 0.0
        if restaurant.rating >= 4.5 and restaurant.user_ratings_total >= 100:
            excellence_boost = 0.8  # Excellent restaurant
        elif restaurant.rating >= 4.2:
            excellence_boost = 0.5  # Very good restaurant

        # User history boost (20% of boost score)
        history_boost = 0.0
        if user_preferences and restaurant.place_id in user_preferences.favorite_restaurants:
            history_boost = 1.0  # User has favorited this place
        elif restaurant.place_id in collaborative_restaurants:
            history_boost = 0.6  # Similar users like this place

        # Novelty/Discovery boost (10% of boost score)
        novelty_boost = 0.0
        if restaurant.user_ratings_total < 100 and restaurant.rating >= 4.0:
            novelty_boost = 0.3  # Hidden gem

        # Combine boost components
        total_boost = (
                trending_boost * 0.40 +
                excellence_boost * 0.30 +
                history_boost * 0.20 +
                novelty_boost * 0.10
        )

        return min(total_boost, 1.0)

    def _restaurant_has_feature(self, restaurant: Restaurant, feature: str) -> bool:
        """Check if restaurant has a specific feature"""

        feature_mapping = {
            "outdoor_seating": restaurant.features.outdoor_seating,
            "live_music": restaurant.features.live_music,
            "parking": restaurant.features.parking_available,
            "wifi": restaurant.features.wifi,
            "delivery": restaurant.features.delivery_available,
            "takeout": restaurant.features.takeout_available,
            "reservations": restaurant.features.accepts_reservations,
            "wheelchair_accessible": restaurant.features.wheelchair_accessible
        }

        return feature_mapping.get(feature, False)

    def _generate_recommendation_reasons(self,
                                         restaurant: Restaurant,
                                         score_breakdown: ScoreBreakdown,
                                         parsed_query: ParsedQuery) -> List[RecommendationReason]:
        """Generate list of reasons why this restaurant is recommended"""

        reasons = []

        # Cuisine match
        if score_breakdown.cuisine_match >= 0.8:
            reasons.append(RecommendationReason.CUISINE_MATCH)

        # Price match
        if score_breakdown.price_match >= 0.8:
            reasons.append(RecommendationReason.PRICE_MATCH)

        # High rating
        if restaurant.rating >= 4.2:
            reasons.append(RecommendationReason.HIGHLY_RATED)

        # Popular choice
        if restaurant.user_ratings_total >= 200:
            reasons.append(RecommendationReason.POPULAR_CHOICE)

        # Feature match
        if score_breakdown.feature_match >= 0.8:
            reasons.append(RecommendationReason.HAS_REQUIRED_FEATURES)

        # Party size appropriate
        party_size = parsed_query.social_context.party_size
        if party_size > 4 and restaurant.features.good_for_groups:
            reasons.append(RecommendationReason.GOOD_FOR_PARTY_SIZE)

        # Collaborative filtering
        if score_breakdown.collaborative_score >= 0.6:
            reasons.append(RecommendationReason.SIMILAR_USERS_LIKED)

        # Time appropriate
        if score_breakdown.time_appropriateness >= 0.9:
            reasons.append(RecommendationReason.LOCATION_CONVENIENT)

        # Occasion match
        if score_breakdown.occasion_match >= 0.8:
            reasons.append(RecommendationReason.MATCHES_OCCASION)

        # Default reason if none specific
        if not reasons:
            if restaurant.rating >= 4.0:
                reasons.append(RecommendationReason.HIGHLY_RATED)
            else:
                reasons.append(RecommendationReason.POPULAR_CHOICE)

        return reasons[:3]  # Limit to top 3 reasons

    def _generate_explanation(self,
                              restaurant: Restaurant,
                              score_breakdown: ScoreBreakdown,
                              reasons: List[RecommendationReason]) -> str:
        """Generate human-readable explanation"""

        explanations = []

        # Start with restaurant basics
        rating_text = f"rated {restaurant.rating}/5.0"
        if restaurant.user_ratings_total >= 100:
            rating_text += f" by {restaurant.user_ratings_total}+ customers"

        explanations.append(f"{restaurant.name} is {rating_text}")

        # Add reason-based explanations
        reason_texts = {
            RecommendationReason.CUISINE_MATCH: f"matches your {restaurant.primary_category.value} preference",
            RecommendationReason.PRICE_MATCH: f"fits your budget ({restaurant.price_level.symbol if restaurant.price_level else 'affordable'})",
            RecommendationReason.HIGHLY_RATED: "has excellent reviews",
            RecommendationReason.POPULAR_CHOICE: "is a popular choice",
            RecommendationReason.HAS_REQUIRED_FEATURES: "has the features you requested",
            RecommendationReason.GOOD_FOR_PARTY_SIZE: "is great for groups",
            RecommendationReason.SIMILAR_USERS_LIKED: "is loved by users with similar taste",
            RecommendationReason.LOCATION_CONVENIENT: "is conveniently located",
            RecommendationReason.MATCHES_OCCASION: "is perfect for the occasion"
        }

        for reason in reasons[:2]:  # Top 2 reasons
            if reason in reason_texts:
                explanations.append(reason_texts[reason])

        # Add confidence indicator
        if score_breakdown.total_score >= 0.8:
            explanations.append("making it a top recommendation")
        elif score_breakdown.total_score >= 0.6:
            explanations.append("making it a solid choice")

        return " and ".join(explanations) + "."

    def _calculate_confidence(self,
                              score_breakdown: ScoreBreakdown,
                              user_preferences: Optional[UserPreferences]) -> float:
        """Calculate confidence in this recommendation"""

        confidence = 0.5  # Base confidence

        # Boost confidence based on score quality
        if score_breakdown.total_score >= 0.8:
            confidence += 0.3
        elif score_breakdown.total_score >= 0.6:
            confidence += 0.2

        # Boost confidence if we have user preferences
        if user_preferences and user_preferences.confidence_score > 0.5:
            confidence += 0.2

        # Boost confidence for high-quality restaurants
        if score_breakdown.rating_score >= 0.8:
            confidence += 0.1

        # Boost confidence for collaborative signals
        if score_breakdown.collaborative_score >= 0.6:
            confidence += 0.1

        return min(confidence, 1.0)

    def _apply_diversity_boost(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Apply diversity boost to ensure variety in recommendations"""

        if len(recommendations) <= 3:
            return recommendations

        # Track cuisines in top recommendations
        cuisine_counts = {}

        for i, rec in enumerate(recommendations):
            cuisine = rec.restaurant.primary_category.value
            cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

            # Apply diversity penalty if too many of same cuisine in top results
            if i < 5 and cuisine_counts[cuisine] > 2:
                # Slightly reduce score to promote diversity
                rec.score.total_score *= 0.95

        # Re-sort after diversity adjustments
        recommendations.sort(key=lambda r: r.score.total_score, reverse=True)

        # Update ranks
        for i, rec in enumerate(recommendations, 1):
            rec.rank = i

        return recommendations

    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get statistics about scoring performance"""

        return {
            "scoring_weights": SCORING_WEIGHTS,
            **self.get_performance_stats()
        }.cuisine_match = cuisine_score
        score_breakdown.price_match = price_score

        return min(total_score, 1.0)

    def _calculate_context_score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Calculate contextual relevance score (30% of total)"""

        total_score = 0.0

        # Time appropriateness (40% of context score)
        time_score = 0.0
        if parsed_query.time_preference.urgency == "now":
            # High priority for open restaurants
            time_score = 1.0 if restaurant.is_open_now else 0.2
        elif parsed_query.time_preference.urgency == "soon":
            time_score = 0.9 if restaurant.is_open_now else 0.6
        else:
            time_score = 0.7  # Default for planning

        # Feature match (35% of context score)
        feature_score = 0.0
        if parsed_query.required_features:
            matched_features = 0
            for feature in parsed_query.required_features:
                if self._restaurant_has_feature(restaurant, feature):
                    matched_features += 1

            feature_score = matched_features / len(parsed_query.required_features)
        else:
            feature_score = 0.5  # Neutral if no specific features required

        # Party size appropriateness (15% of context score)
        party_size_score = 0.0
        party_size = parsed_query.social_context.party_size

        if party_size <= 2:
            party_size_score = 1.0  # Most restaurants handle couples
        elif party_size <= 4:
            party_size_score = 0.9  # Most restaurants handle small groups
        elif party_size <= 8:
            party_size_score = 0.7 if restaurant.features.good_for_groups else 0.5
        else:
            party_size_score = 0.8 if restaurant.features.good_for_groups else 0.3

        # Occasion match (10% of context score)
        occasion_score = 0.0
        occasion = parsed_query.social_context.occasion

        if occasion:
            if "date" in occasion.lower() or "romantic" in occasion.lower():
                occasion_score = 0.9 if restaurant.features.romantic else 0.5
            elif "business" in occasion.lower():
                occasion_score = 0.9 if "upscale" in str(restaurant.features) else 0.6
            elif "family" in occasion.lower():
                occasion_score = 0.9 if restaurant.features.good_for_kids else 0.4
            else:
                occasion_score = 0.7  # Default for other occasions
        else:
            occasion_score = 0.7  # Neutral if no specific occasion

        # Combine context components
        total_score = (
                time_score * 0.40 +
                feature_score * 0.35 +
                party_size_score * 0.15 +
                occasion_score * 0.10
        )

        # Store component scores
        score_breakdown.time_appropriateness = time_score
        score_breakdown.feature_match = feature_score
        score_breakdown.occasion_match = occasion_score

        return min(total_score, 1.0)

    def _calculate_quality_score(self, restaurant: Restaurant) -> float:
        """Calculate restaurant quality score (15% of total)"""

        # Rating component (70% of quality score)
        rating_score = 0.0
        if restaurant.rating > 0:
            # Normalize 1-5 rating to 0-1, with 4.0+ being excellent
            rating_score = min((restaurant.rating - 1) / 4, 1.0)

            # Boost for high ratings
            if restaurant.rating >= 4.5:
                rating_score = min(rating_score * 1.1, 1.0)

        # Popularity component (30% of quality score)
        popularity_score = 0.0
        if restaurant.user_ratings_total > 0:
            # Log scale for review count (diminishing returns)
            popularity_score = min(math.log(restaurant.user_ratings_total + 1) / math.log(1000), 1.0)

        # Combine quality components
        total_score = (
                rating_score * 0.70 +
                popularity_score * 0.30
        )

        # Store component scores
        score_breakdown