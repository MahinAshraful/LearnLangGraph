from typing import Dict, Any, List
import logging

from .base_node import BaseNode
from ..recommendation_state import RecommendationState
from ...models.recommendation import Recommendation

logger = logging.getLogger(__name__)


class OutputFormatterNode(BaseNode):
    """Formats final recommendations into user-friendly response"""

    def __init__(self):
        super().__init__("output_formatter")

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Format final recommendations for user presentation"""

        final_recommendations = state.get("final_recommendations", [])
        scored_recommendations = state.get("scored_recommendations", [])
        parsed_query = state.get("parsed_query")
        confidence_score = state.get("confidence_score", 0.0)

        # Use scored recommendations if no final recommendations
        if not final_recommendations and scored_recommendations:
            final_recommendations = scored_recommendations[:10]  # Top 10

        if not final_recommendations:
            return {
                "response_message": "Sorry, I couldn't find any restaurants matching your criteria. Try broadening your search or exploring a different area.",
                "confidence_score": 0.0
            }

        try:
            # Generate response message
            response_message = self._generate_response_message(
                final_recommendations, parsed_query, confidence_score
            )

            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(
                final_recommendations, confidence_score
            )

            logger.info(f"Formatted {len(final_recommendations)} recommendations")

            return {
                "response_message": response_message,
                "final_recommendations": final_recommendations,
                "confidence_score": final_confidence
            }

        except Exception as e:
            logger.error(f"Output formatting failed: {e}")
            return self._handle_error(state, f"Failed to format output: {str(e)}")

    def _generate_response_message(self,
                                   recommendations: List[Recommendation],
                                   parsed_query,
                                   confidence_score: float) -> str:
        """Generate personalized response message"""

        if not recommendations:
            return "I couldn't find any restaurants matching your preferences."

        top_rec = recommendations[0]
        restaurant = top_rec.restaurant

        # Start with greeting and summary
        response_parts = []

        # Confidence-based opening
        if confidence_score >= 0.8:
            response_parts.append("Great news! I found some excellent options for you.")
        elif confidence_score >= 0.6:
            response_parts.append("I found several good restaurants that match your preferences.")
        else:
            response_parts.append("Here are some restaurant options you might enjoy.")

        # Highlight top recommendation
        top_reason = top_rec.primary_reasons[0].value.replace("_", " ") if top_rec.primary_reasons else "good choice"

        response_parts.append(
            f"My top recommendation is **{restaurant.name}**, "
            f"a {restaurant.primary_category.value} restaurant rated {restaurant.rating}/5.0. "
            f"I suggest it because it {top_reason.lower()}."
        )

        # Add context about the search
        if len(recommendations) > 1:
            response_parts.append(f"I've found {len(recommendations)} total options for you to explore.")

        # Add helpful context
        if parsed_query:
            context_parts = []

            if parsed_query.social_context.party_size > 2:
                context_parts.append(f"great for groups of {parsed_query.social_context.party_size}")

            if parsed_query.time_preference.urgency == "now":
                context_parts.append("currently open")

            if parsed_query.required_features:
                feature_names = [f.replace("_", " ") for f in parsed_query.required_features[:2]]
                context_parts.append(f"with {', '.join(feature_names)}")

            if context_parts:
                response_parts.append(f"All options are {' and '.join(context_parts)}.")

        return " ".join(response_parts)

    def _calculate_final_confidence(self,
                                    recommendations: List[Recommendation],
                                    base_confidence: float) -> float:
        """Calculate final confidence score"""

        if not recommendations:
            return 0.0

        # Average confidence of top 3 recommendations
        top_recs = recommendations[:3]
        avg_confidence = sum(rec.confidence for rec in top_recs) / len(top_recs)

        # Combine with base confidence
        final_confidence = (base_confidence + avg_confidence) / 2

        # Boost if we have high-quality recommendations
        if recommendations[0].score.total_score >= 0.8:
            final_confidence = min(final_confidence * 1.1, 1.0)

        return min(final_confidence, 1.0)

    def _format_recommendation_for_api(self, rec: Recommendation) -> Dict[str, Any]:
        """Format recommendation for API response without exposing 0.0 ratings"""

        restaurant = rec.restaurant

        return {
            "id": str(rec.id),
            "restaurant": {
                "place_id": restaurant.place_id,
                "name": restaurant.name,
                "cuisine": restaurant.primary_category.value,
                "rating": restaurant.rating if restaurant.rating > 0 else None,
                "user_ratings_total": restaurant.user_ratings_total if restaurant.user_ratings_total > 0 else None,
                "price_level": restaurant.price_level.value if restaurant.price_level else None,
                "address": restaurant.formatted_address,
                "phone": restaurant.phone_number,
                "website": restaurant.website,
                "photos": restaurant.photos[:3] if restaurant.photos else [],
                "location": {
                    "lat": restaurant.location.latitude,
                    "lng": restaurant.location.longitude
                },
                "features": restaurant.features.dict() if restaurant.features else {},
                "opening_hours": restaurant.opening_hours.dict() if restaurant.opening_hours else None
            },
            "recommendation": {
                "rank": rec.rank,
                "score": round(rec.total_score, 3),
                "confidence": round(rec.confidence, 3),
                "explanation": rec.explanation,
                "reasons": [reason.value for reason in rec.primary_reasons],
                "novelty": round(rec.novelty_score, 3) if hasattr(rec, 'novelty_score') else 0.0
            },
            "score_breakdown": {
                "preference_match": round(rec.score.preference_score, 3),
                "context_relevance": round(rec.score.context_score, 3),
                "quality": round(rec.score.quality_score, 3),
                "boost": round(rec.score.boost_score, 3)
            }
        }

    def _format_recommendation_for_display(self, rec: Recommendation) -> str:
        """Format recommendation for console display without showing 0.0 ratings"""

        restaurant = rec.restaurant

        # Format rating - hide if 0.0
        rating_str = f"⭐ {restaurant.rating}/5" if restaurant.rating > 0 else "⭐ New"

        # Format price - hide if None
        price_str = restaurant.price_level.symbol if restaurant.price_level else "Price varies"

        return (f"  {rec.rank}. {restaurant.name} ({restaurant.primary_category.value})\n"
                f"     {rating_str} | {price_str} | Score: {rec.score.total_score:.3f}\n"
                f"     💡 {rec.explanation}")