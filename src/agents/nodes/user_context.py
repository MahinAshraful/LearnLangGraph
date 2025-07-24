from typing import Dict, Any, List
import logging

from .base_node import BaseNode
from ..state.recommendation_state import RecommendationState
from ...infrastructure.databases.vector_db.base import VectorDBAdapter
from ...domain.models.user import UserPreferences, User
from ...domain.models.query import QueryContext
from ...domain.models.common import Location

logger = logging.getLogger(__name__)


class UserContextNode(BaseNode):
    """Retrieves user preferences and finds similar users for collaborative filtering"""

    def __init__(self, vector_db: VectorDBAdapter):
        super().__init__("user_context")
        self.vector_db = vector_db

    async def execute(self, state: RecommendationState) -> Dict[str, Any]:
        """Retrieve user context including preferences and similar users"""

        user_id = state.get("user_id")
        parsed_query = state.get("parsed_query")

        if not user_id:
            return self._handle_error(state, "No user ID provided", is_fatal=True)

        try:
            # Retrieve user preferences
            user_preferences = await self._get_user_preferences(user_id)

            # Find similar users for collaborative filtering
            similar_users = await self._find_similar_users(user_id, user_preferences)

            # Get restaurants liked by similar users
            collaborative_restaurants = await self._get_collaborative_recommendations(similar_users)

            # Create query context
            query_context = self._create_query_context(state, user_preferences)

            cache_hits = 1 if user_preferences else 0  # Assume cache hit if user found

            logger.info(
                f"Retrieved context for user {user_id}: {len(similar_users)} similar users, {len(collaborative_restaurants)} collaborative recommendations")

            return {
                "user_preferences": user_preferences,
                "similar_users": similar_users,
                "collaborative_restaurants": collaborative_restaurants,
                "query_context": query_context,
                **self._update_performance_tracking(state, cache_hits=cache_hits)
            }

        except Exception as e:
            logger.error(f"User context retrieval failed: {e}")
            # Non-fatal error - we can proceed without user context
            return {
                **self._handle_error(state, f"Could not retrieve user context: {str(e)}"),
                "user_preferences": None,
                "similar_users": [],
                "collaborative_restaurants": []
            }

    async def _get_user_preferences(self, user_id: str) -> UserPreferences:
        """Retrieve user preferences from vector database"""

        try:
            # Get user document from vector DB
            user_doc = await self.vector_db.get_document(user_id, "user_preferences")

            if user_doc:
                # Extract preferences from metadata
                metadata = user_doc.metadata

                # Create UserPreferences from metadata
                user_preferences = UserPreferences(
                    user_id=user_id,
                    favorite_cuisines=[],
                    preferred_price_levels=[],
                    dietary_restrictions=[],
                    preferred_ambiance=[],
                    confidence_score=0.8  # Assume good confidence for existing users
                )

                # Populate from metadata
                if "preferred_cuisines" in metadata:
                    from ...domain.models.restaurant import RestaurantCategory
                    user_preferences.favorite_cuisines = [
                        RestaurantCategory(cuisine) for cuisine in metadata["preferred_cuisines"]
                    ]

                if "price_preference" in metadata:
                    from ...domain.models.restaurant import PriceLevel
                    try:
                        user_preferences.preferred_price_levels = [PriceLevel(metadata["price_preference"])]
                    except (ValueError, KeyError):
                        pass

                if "ambiance_preferences" in metadata:
                    from ...domain.models.user import AmbiancePreference
                    user_preferences.preferred_ambiance = [
                        AmbiancePreference(amb) for amb in metadata["ambiance_preferences"]
                        if amb in [a.value for a in AmbiancePreference]
                    ]

                if "dietary_restrictions" in metadata:
                    from ...domain.models.user import DietaryRestriction
                    user_preferences.dietary_restrictions = [
                        DietaryRestriction(diet) for diet in metadata["dietary_restrictions"]
                        if diet in [d.value for d in DietaryRestriction]
                    ]

                logger.debug(f"Found existing preferences for user {user_id}")
                return user_preferences

            else:
                # New user - create default preferences
                logger.info(f"Creating default preferences for new user {user_id}")
                return UserPreferences(
                    user_id=user_id,
                    confidence_score=0.0  # No confidence for new user
                )

        except Exception as e:
            logger.warning(f"Failed to retrieve user preferences for {user_id}: {e}")
            # Return default preferences
            return UserPreferences(
                user_id=user_id,
                confidence_score=0.0
            )

    async def _find_similar_users(self, user_id: str, user_preferences: UserPreferences) -> List[UserPreferences]:
        """Find users with similar preferences using vector similarity"""

        try:
            # For mock vector DB, use the specialized method
            if hasattr(self.vector_db, 'get_similar_users'):
                similar_results = await self.vector_db.get_similar_users(user_id, limit=5)

                similar_users = []
                for result in similar_results:
                    # Convert vector document metadata to UserPreferences
                    metadata = result.document.metadata

                    similar_user = UserPreferences(
                        user_id=metadata.get("user_id", "unknown"),
                        confidence_score=result.score
                    )

                    # Populate preferences from metadata
                    if "preferred_cuisines" in metadata:
                        from ...domain.models.restaurant import RestaurantCategory
                        similar_user.favorite_cuisines = [
                            RestaurantCategory(cuisine) for cuisine in metadata["preferred_cuisines"]
                        ]

                    if "price_preference" in metadata:
                        from ...domain.models.restaurant import PriceLevel
                        try:
                            similar_user.preferred_price_levels = [PriceLevel(metadata["price_preference"])]
                        except (ValueError, KeyError):
                            pass

                    similar_users.append(similar_user)

                logger.debug(f"Found {len(similar_users)} similar users for {user_id}")
                return similar_users

            else:
                # Fallback for production vector DB
                # Would need user's embedding to search
                logger.warning("Vector DB doesn't support similar user search")
                return []

        except Exception as e:
            logger.warning(f"Failed to find similar users for {user_id}: {e}")
            return []

    async def _get_collaborative_recommendations(self, similar_users: List[UserPreferences]) -> List[str]:
        """Get restaurant IDs that similar users have liked"""

        if not similar_users:
            return []

        try:
            # For mock vector DB, use specialized method
            if hasattr(self.vector_db, 'get_restaurants_liked_by_similar_users'):
                # Use the first similar user as representative
                first_user = similar_users[0]
                liked_restaurants = await self.vector_db.get_restaurants_liked_by_similar_users(
                    first_user.user_id,
                    limit=10
                )

                logger.debug(f"Found {len(liked_restaurants)} collaborative recommendations")
                return liked_restaurants

            else:
                # Fallback - would query user activity data in production
                logger.warning("Vector DB doesn't support collaborative recommendations")
                return []

        except Exception as e:
            logger.warning(f"Failed to get collaborative recommendations: {e}")
            return []

    def _create_query_context(self, state: RecommendationState, user_preferences: UserPreferences) -> QueryContext:
        """Create query context from current state"""

        from datetime import datetime

        parsed_query = state.get("parsed_query")
        user_id = state.get("user_id", "")
        user_location = state.get("user_location")

        # Generate session ID (simplified)
        import uuid
        session_id = str(uuid.uuid4())[:8]

        query_context = QueryContext(
            user_id=user_id,
            session_id=session_id,
            device_type="web",  # Default
            user_location=user_location,
            time_of_query=datetime.utcnow(),
            use_personalization=user_preferences.confidence_score > 0.1,
            include_social_signals=len(state.get("similar_users", [])) > 0
        )

        # Add query to history
        if parsed_query:
            query_context.add_to_history(parsed_query.original_query, 0)  # Will update result count later

        return query_context

    async def _update_user_preferences_from_query(self, user_preferences: UserPreferences,
                                                  parsed_query) -> UserPreferences:
        """Update user preferences based on current query (implicit feedback)"""

        if not parsed_query:
            return user_preferences

        # Add cuisine preferences from query
        for cuisine in parsed_query.cuisine_preferences:
            if cuisine not in user_preferences.favorite_cuisines:
                user_preferences.favorite_cuisines.append(cuisine)
                user_preferences.add_cuisine_preference(cuisine, 0.5)  # Implicit signal

        # Add price preferences
        for price in parsed_query.price_preferences:
            if price not in user_preferences.preferred_price_levels:
                user_preferences.preferred_price_levels.append(price)

        # Add dietary restrictions
        for dietary in parsed_query.dietary_requirements:
            if dietary not in user_preferences.dietary_restrictions:
                user_preferences.dietary_restrictions.append(dietary)

        # Add ambiance preferences
        for ambiance in parsed_query.ambiance_preferences:
            if ambiance not in user_preferences.preferred_ambiance:
                user_preferences.preferred_ambiance.append(ambiance)

        # Update confidence slightly
        user_preferences.confidence_score = min(user_preferences.confidence_score + 0.1, 1.0)

        return user_preferences

    async def save_updated_preferences(self, user_preferences: UserPreferences):
        """Save updated user preferences back to vector database"""

        try:
            # Create metadata from preferences
            metadata = {
                "user_id": user_preferences.user_id,
                "preferred_cuisines": [c.value for c in user_preferences.favorite_cuisines],
                "price_preference": user_preferences.preferred_price_levels[
                    0].value if user_preferences.preferred_price_levels else None,
                "ambiance_preferences": [a.value for a in user_preferences.preferred_ambiance],
                "dietary_restrictions": [d.value for d in user_preferences.dietary_restrictions],
                "confidence_score": user_preferences.confidence_score,
                "last_updated": user_preferences.last_updated.isoformat()
            }

            # Create or update vector document
            # In production, this would generate an embedding from the preferences
            from ...infrastructure.databases.vector_db.base import VectorDocument
            import numpy as np

            # Generate simple embedding (in production, use OpenAI embeddings)
            embedding = np.random.normal(0, 0.1, 1536).tolist()  # Placeholder

            document = VectorDocument(
                id=user_preferences.user_id,
                embedding=embedding,
                metadata=metadata,
                content=f"User preferences for {user_preferences.user_id}"
            )

            # Save to vector database
            success = await self.vector_db.update_document(document, "user_preferences")

            if success:
                logger.debug(f"Updated preferences for user {user_preferences.user_id}")
            else:
                logger.warning(f"Failed to update preferences for user {user_preferences.user_id}")

        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")


class UserPreferenceTracker:
    """Helper class to track and learn user preferences over time"""

    def __init__(self, vector_db: VectorDBAdapter):
        self.vector_db = vector_db

    async def learn_from_interaction(self, user_id: str, restaurant_id: str,
                                     interaction_type: str, rating: float = None):
        """Learn from user interactions with restaurants"""

        try:
            # Get current user preferences
            user_doc = await self.vector_db.get_document(user_id, "user_preferences")

            if user_doc:
                metadata = user_doc.metadata.copy()

                # Update preferences based on interaction
                if interaction_type in ["like", "book", "visit"] and rating and rating >= 4.0:
                    # Positive signal - strengthen preferences
                    confidence = metadata.get("confidence_score", 0.0)
                    metadata["confidence_score"] = min(confidence + 0.1, 1.0)

                elif interaction_type in ["dislike"] and rating and rating <= 2.0:
                    # Negative signal - might indicate preference against this type
                    pass  # Would implement negative preference tracking

                # Update document
                from ...infrastructure.databases.vector_db.base import VectorDocument
                updated_doc = VectorDocument(
                    id=user_id,
                    embedding=user_doc.embedding,  # Keep existing embedding
                    metadata=metadata,
                    content=user_doc.content
                )

                await self.vector_db.update_document(updated_doc, "user_preferences")

                logger.debug(f"Updated preferences for user {user_id} based on {interaction_type}")

        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")

    async def get_preference_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's dining preferences"""

        try:
            user_doc = await self.vector_db.get_document(user_id, "user_preferences")

            if not user_doc:
                return {"status": "no_data", "insights": []}

            metadata = user_doc.metadata
            insights = []

            # Cuisine insights
            if "preferred_cuisines" in metadata and metadata["preferred_cuisines"]:
                top_cuisine = metadata["preferred_cuisines"][0]
                insights.append(f"You seem to enjoy {top_cuisine} cuisine")

            # Price insights
            if "price_preference" in metadata:
                price_level = metadata["price_preference"]
                price_labels = {1: "budget-friendly", 2: "moderately-priced", 3: "upscale", 4: "fine dining"}
                label = price_labels.get(price_level, "various price ranges")
                insights.append(f"You typically prefer {label} restaurants")

            # Confidence insights
            confidence = metadata.get("confidence_score", 0.0)
            if confidence > 0.7:
                insights.append("We have high confidence in your preferences")
            elif confidence > 0.3:
                insights.append("We're learning your preferences")
            else:
                insights.append("We'd love to learn more about your dining preferences")

            return {
                "status": "success",
                "confidence_score": confidence,
                "insights": insights,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Failed to get preference insights: {e}")
            return {"status": "error", "error": str(e)}