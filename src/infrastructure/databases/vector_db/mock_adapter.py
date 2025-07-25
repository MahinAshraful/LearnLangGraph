import numpy as np
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio

from .base import VectorDBAdapter, VectorDocument, VectorSearchResult
from ....models.user import UserPreferences, DietaryRestriction, AmbiancePreference
from ....models.restaurant import RestaurantCategory, PriceLevel

logger = logging.getLogger(__name__)


class MockVectorAdapter(VectorDBAdapter):
    """Mock vector database adapter for development and testing"""

    def __init__(self, collection_name: str = "restaurant_recommendations"):
        super().__init__(collection_name)

        # In-memory storage
        self.collections: Dict[str, Dict[str, VectorDocument]] = {}
        self.collection_metadata: Dict[str, Dict[str, Any]] = {}

        # Initialize with mock data
        self._initialize_mock_data()

    async def connect(self) -> bool:
        """Mock connection - always succeeds"""
        await asyncio.sleep(0.01)  # Simulate connection delay
        self.is_connected = True
        logger.info("Connected to Mock Vector Database")
        return True

    async def disconnect(self):
        """Mock disconnection"""
        self.is_connected = False
        logger.info("Disconnected from Mock Vector Database")

    async def create_collection(self, collection_name: str, dimension: int,
                                metadata_schema: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection"""
        self.collections[collection_name] = {}
        self.collection_metadata[collection_name] = {
            "dimension": dimension,
            "created_at": datetime.utcnow(),
            "schema": metadata_schema or {}
        }
        logger.info(f"Created mock collection: {collection_name} (dim: {dimension})")
        return True

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        if collection_name in self.collections:
            del self.collections[collection_name]
            del self.collection_metadata[collection_name]
            logger.info(f"Deleted mock collection: {collection_name}")
            return True
        return False

    async def list_collections(self) -> List[str]:
        """List all collections"""
        return list(self.collections.keys())

    async def insert_documents(self, documents: List[VectorDocument],
                               collection_name: Optional[str] = None) -> bool:
        """Insert multiple documents"""
        target_collection = collection_name or self.collection_name

        # Create collection if it doesn't exist
        if target_collection not in self.collections:
            dimension = len(documents[0].embedding) if documents else 1536
            await self.create_collection(target_collection, dimension)

        # Insert documents
        for doc in documents:
            self.collections[target_collection][doc.id] = doc

        logger.debug(f"Inserted {len(documents)} documents into {target_collection}")
        return True

    async def insert_document(self, document: VectorDocument,
                              collection_name: Optional[str] = None) -> bool:
        """Insert a single document"""
        return await self.insert_documents([document], collection_name)

    async def search_similar(self, query_embedding: List[float],
                             limit: int = 10,
                             threshold: Optional[float] = None,
                             metadata_filter: Optional[Dict[str, Any]] = None,
                             collection_name: Optional[str] = None) -> List[VectorSearchResult]:
        """Search for similar vectors using cosine similarity"""

        target_collection = collection_name or self.collection_name

        if target_collection not in self.collections:
            return []

        collection = self.collections[target_collection]
        results = []

        # Calculate similarities
        query_vector = np.array(query_embedding)

        for doc_id, document in collection.items():
            # Apply metadata filter if specified
            if metadata_filter and not self._matches_filter(document.metadata, metadata_filter):
                continue

            # Calculate cosine similarity
            doc_vector = np.array(document.embedding)

            # Normalize vectors
            query_norm = np.linalg.norm(query_vector)
            doc_norm = np.linalg.norm(doc_vector)

            if query_norm == 0 or doc_norm == 0:
                similarity = 0.0
            else:
                cosine_sim = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
                similarity = float(cosine_sim)

            # Convert to distance (0 = identical, higher = more different)
            distance = 1.0 - similarity

            # Apply threshold filter
            if threshold is not None and similarity < threshold:
                continue

            result = VectorSearchResult(
                document=document,
                score=similarity,
                distance=distance
            )
            results.append(result)

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        # Limit results
        results = results[:limit]

        logger.debug(f"Found {len(results)} similar documents in {target_collection}")
        return results

    async def get_document(self, document_id: str,
                           collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        """Get a document by ID"""
        target_collection = collection_name or self.collection_name

        if target_collection in self.collections:
            return self.collections[target_collection].get(document_id)

        return None

    async def update_document(self, document: VectorDocument,
                              collection_name: Optional[str] = None) -> bool:
        """Update an existing document"""
        target_collection = collection_name or self.collection_name

        if target_collection in self.collections:
            self.collections[target_collection][document.id] = document
            logger.debug(f"Updated document {document.id} in {target_collection}")
            return True

        return False

    async def delete_document(self, document_id: str,
                              collection_name: Optional[str] = None) -> bool:
        """Delete a document by ID"""
        target_collection = collection_name or self.collection_name

        if target_collection in self.collections:
            if document_id in self.collections[target_collection]:
                del self.collections[target_collection][document_id]
                logger.debug(f"Deleted document {document_id} from {target_collection}")
                return True

        return False

    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """Count documents in collection"""
        target_collection = collection_name or self.collection_name

        if target_collection in self.collections:
            return len(self.collections[target_collection])

        return 0

    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, expected_value in filter_criteria.items():
            if key not in metadata:
                return False

            actual_value = metadata[key]

            # Handle different comparison types
            if isinstance(expected_value, dict):
                # Range or complex filters (e.g., {"$gte": 3.5})
                for op, value in expected_value.items():
                    if op == "$gte" and actual_value < value:
                        return False
                    elif op == "$lte" and actual_value > value:
                        return False
                    elif op == "$eq" and actual_value != value:
                        return False
                    elif op == "$ne" and actual_value == value:
                        return False
                    elif op == "$in" and actual_value not in value:
                        return False
                    elif op == "$nin" and actual_value in value:
                        return False
            elif isinstance(expected_value, list):
                # Any of these values
                if actual_value not in expected_value:
                    return False
            else:
                # Exact match
                if actual_value != expected_value:
                    return False

        return True

    def _initialize_mock_data(self):
        """Initialize with realistic mock user preference embeddings"""

        # Create user preferences collection
        self.collections["user_preferences"] = {}
        self.collection_metadata["user_preferences"] = {
            "dimension": 1536,
            "created_at": datetime.utcnow(),
            "schema": {"user_id": "string", "preferences_hash": "string"}
        }

        # Generate mock user embeddings
        user_archetypes = [
            {
                "user_id": "foodie_explorer_0",
                "cuisines": [RestaurantCategory.JAPANESE, RestaurantCategory.THAI, RestaurantCategory.INDIAN],
                "price_pref": PriceLevel.EXPENSIVE,
                "ambiance": [AmbiancePreference.TRENDY, AmbiancePreference.AUTHENTIC],
                "dietary": []
            },
            {
                "user_id": "budget_conscious_0",
                "cuisines": [RestaurantCategory.AMERICAN, RestaurantCategory.MEXICAN, RestaurantCategory.ITALIAN],
                "price_pref": PriceLevel.MODERATE,
                "ambiance": [AmbiancePreference.CASUAL, AmbiancePreference.FAMILY_FRIENDLY],
                "dietary": []
            },
            {
                "user_id": "health_focused_0",
                "cuisines": [RestaurantCategory.MEDITERRANEAN, RestaurantCategory.VEGETARIAN],
                "price_pref": PriceLevel.MODERATE,
                "ambiance": [AmbiancePreference.QUIET, AmbiancePreference.COZY],
                "dietary": [DietaryRestriction.VEGETARIAN, DietaryRestriction.GLUTEN_FREE]
            },
            {
                "user_id": "business_diner_0",
                "cuisines": [RestaurantCategory.AMERICAN, RestaurantCategory.FRENCH, RestaurantCategory.ITALIAN],
                "price_pref": PriceLevel.VERY_EXPENSIVE,
                "ambiance": [AmbiancePreference.UPSCALE, AmbiancePreference.QUIET, AmbiancePreference.BUSINESS],
                "dietary": []
            },
            {
                "user_id": "comfort_seeker_0",
                "cuisines": [RestaurantCategory.AMERICAN, RestaurantCategory.ITALIAN],
                "price_pref": PriceLevel.MODERATE,
                "ambiance": [AmbiancePreference.COZY, AmbiancePreference.TRADITIONAL],
                "dietary": []
            }
        ]

        for i, archetype in enumerate(user_archetypes):
            # Generate embedding based on user preferences
            embedding = self._generate_user_embedding(archetype)

            # Create multiple similar users for each archetype
            for j in range(3):
                user_id = f"{archetype['user_id']}_{j}"

                # Add some variation to the embedding
                varied_embedding = self._add_embedding_variation(embedding, variation=0.1)

                metadata = {
                    "user_id": user_id,
                    "archetype": archetype["user_id"],
                    "preferred_cuisines": [c.value for c in archetype["cuisines"]],
                    "price_preference": archetype["price_pref"].value,
                    "ambiance_preferences": [a.value for a in archetype["ambiance"]],
                    "dietary_restrictions": [d.value for d in archetype["dietary"]],
                    "last_updated": datetime.utcnow().isoformat()
                }

                document = VectorDocument(
                    id=user_id,
                    embedding=varied_embedding,
                    metadata=metadata,
                    content=f"User preferences for {user_id}"
                )

                self.collections["user_preferences"][user_id] = document

        # Create restaurant features collection
        self.collections["restaurant_features"] = {}
        self.collection_metadata["restaurant_features"] = {
            "dimension": 1536,
            "created_at": datetime.utcnow(),
            "schema": {"place_id": "string", "cuisine": "string", "price_level": "int"}
        }

        # Generate mock restaurant embeddings
        restaurant_templates = [
            {"place_id": "mock_place_0", "name": "Mario's Authentic Italian", "cuisine": RestaurantCategory.ITALIAN,
             "price": PriceLevel.EXPENSIVE},
            {"place_id": "mock_place_1", "name": "Golden Dragon", "cuisine": RestaurantCategory.CHINESE,
             "price": PriceLevel.MODERATE},
            {"place_id": "mock_place_2", "name": "Sakura Sushi", "cuisine": RestaurantCategory.JAPANESE,
             "price": PriceLevel.EXPENSIVE},
            {"place_id": "mock_place_3", "name": "Thai Spice", "cuisine": RestaurantCategory.THAI,
             "price": PriceLevel.MODERATE},
            {"place_id": "mock_place_4", "name": "The Burger Joint", "cuisine": RestaurantCategory.AMERICAN,
             "price": PriceLevel.MODERATE},
            {"place_id": "mock_place_5", "name": "Mediterranean Breeze", "cuisine": RestaurantCategory.MEDITERRANEAN,
             "price": PriceLevel.MODERATE},
            {"place_id": "mock_place_6", "name": "El Mariachi", "cuisine": RestaurantCategory.MEXICAN,
             "price": PriceLevel.MODERATE},
            {"place_id": "mock_place_7", "name": "Café Parisien", "cuisine": RestaurantCategory.FRENCH,
             "price": PriceLevel.EXPENSIVE},
            {"place_id": "mock_place_8", "name": "Curry Palace", "cuisine": RestaurantCategory.INDIAN,
             "price": PriceLevel.MODERATE},
            {"place_id": "mock_place_9", "name": "Steakhouse 21", "cuisine": RestaurantCategory.STEAKHOUSE,
             "price": PriceLevel.VERY_EXPENSIVE},
        ]

        for template in restaurant_templates:
            embedding = self._generate_restaurant_embedding(template)

            metadata = {
                "place_id": template["place_id"],
                "name": template["name"],
                "cuisine": template["cuisine"].value,
                "price_level": template["price"].value,
                "last_updated": datetime.utcnow().isoformat()
            }

            document = VectorDocument(
                id=template["place_id"],
                embedding=embedding,
                metadata=metadata,
                content=f"Restaurant features for {template['name']}"
            )

            self.collections["restaurant_features"][template["place_id"]] = document

        logger.info(
            f"Initialized mock vector DB with {len(self.collections['user_preferences'])} users and {len(self.collections['restaurant_features'])} restaurants")

    def _generate_user_embedding(self, archetype: Dict[str, Any]) -> List[float]:
        """Generate a realistic user preference embedding"""

        # Create base embedding with some randomness
        np.random.seed(hash(archetype["user_id"]) % 2 ** 32)
        base_embedding = np.random.normal(0, 0.1, 1536)

        # Add cuisine preferences (stronger signals)
        cuisine_weights = {
            RestaurantCategory.ITALIAN: 0.8,
            RestaurantCategory.JAPANESE: 0.9,
            RestaurantCategory.THAI: 0.7,
            RestaurantCategory.INDIAN: 0.6,
            RestaurantCategory.AMERICAN: 0.5,
            RestaurantCategory.MEXICAN: 0.6,
            RestaurantCategory.FRENCH: 0.8,
            RestaurantCategory.CHINESE: 0.6,
            RestaurantCategory.MEDITERRANEAN: 0.7,
            RestaurantCategory.VEGETARIAN: 0.9
        }

        # Boost embedding dimensions for preferred cuisines
        for i, cuisine in enumerate(archetype["cuisines"]):
            start_idx = (i * 150) % 1536
            end_idx = min(start_idx + 100, 1536)
            weight = cuisine_weights.get(cuisine, 0.5)
            base_embedding[start_idx:end_idx] += np.random.normal(weight, 0.1, end_idx - start_idx)

        # Add price preference signal
        price_start = 200
        price_weight = archetype["price_pref"].value * 0.3
        base_embedding[price_start:price_start + 50] += np.random.normal(price_weight, 0.05, 50)

        # Add ambiance preferences
        ambiance_start = 300
        for i, ambiance in enumerate(archetype["ambiance"]):
            amb_start = ambiance_start + (i * 30)
            amb_end = min(amb_start + 30, 1536)
            base_embedding[amb_start:amb_end] += np.random.normal(0.4, 0.05, amb_end - amb_start)

        # Add dietary restrictions signal
        if archetype["dietary"]:
            dietary_start = 500
            for restriction in archetype["dietary"]:
                base_embedding[dietary_start:dietary_start + 20] += np.random.normal(0.6, 0.1, 20)
                dietary_start += 20

        # Normalize to unit vector (common for embeddings)
        norm = np.linalg.norm(base_embedding)
        if norm > 0:
            base_embedding = base_embedding / norm

        return base_embedding.tolist()

    def _generate_restaurant_embedding(self, template: Dict[str, Any]) -> List[float]:
        """Generate a restaurant feature embedding"""

        # Create base embedding
        np.random.seed(hash(template["place_id"]) % 2 ** 32)
        base_embedding = np.random.normal(0, 0.1, 1536)

        # Add cuisine signal (should align with user cuisine preferences)
        cuisine_map = {
            RestaurantCategory.ITALIAN: 0,
            RestaurantCategory.JAPANESE: 1,
            RestaurantCategory.THAI: 2,
            RestaurantCategory.INDIAN: 3,
            RestaurantCategory.AMERICAN: 4,
            RestaurantCategory.MEXICAN: 5,
            RestaurantCategory.FRENCH: 6,
            RestaurantCategory.CHINESE: 7,
            RestaurantCategory.MEDITERRANEAN: 8,
            RestaurantCategory.VEGETARIAN: 9
        }

        cuisine_idx = cuisine_map.get(template["cuisine"], 0)
        start_idx = (cuisine_idx * 150) % 1536
        end_idx = min(start_idx + 100, 1536)
        base_embedding[start_idx:end_idx] += np.random.normal(0.8, 0.1, end_idx - start_idx)

        # Add price level signal
        price_start = 200
        price_weight = template["price"].value * 0.3
        base_embedding[price_start:price_start + 50] += np.random.normal(price_weight, 0.05, 50)

        # Add some restaurant-specific features
        quality_start = 600
        base_embedding[quality_start:quality_start + 50] += np.random.normal(0.5, 0.1, 50)

        # Normalize
        norm = np.linalg.norm(base_embedding)
        if norm > 0:
            base_embedding = base_embedding / norm

        return base_embedding.tolist()

    def _add_embedding_variation(self, embedding: List[float], variation: float = 0.1) -> List[float]:
        """Add random variation to an embedding"""

        embedding_array = np.array(embedding)
        noise = np.random.normal(0, variation, len(embedding))
        varied_embedding = embedding_array + noise

        # Normalize
        norm = np.linalg.norm(varied_embedding)
        if norm > 0:
            varied_embedding = varied_embedding / norm

        return varied_embedding.tolist()

    async def get_similar_users(self, user_id: str, limit: int = 5) -> List[VectorSearchResult]:
        """Get users with similar preferences"""

        # Get the user's embedding
        user_doc = await self.get_document(user_id, "user_preferences")
        if not user_doc:
            return []

        # Search for similar users (exclude self)
        results = await self.search_similar(
            query_embedding=user_doc.embedding,
            limit=limit + 1,  # +1 to account for self
            collection_name="user_preferences"
        )

        # Filter out the user themselves
        filtered_results = [r for r in results if r.document.id != user_id]

        return filtered_results[:limit]

    async def get_restaurants_liked_by_similar_users(self, user_id: str, limit: int = 10) -> List[str]:
        """Get restaurant IDs liked by users with similar preferences"""

        similar_users = await self.get_similar_users(user_id, limit=5)

        # For demo purposes, return mock restaurant preferences
        # In production, this would query actual user activity data
        restaurant_preferences = []

        for user_result in similar_users:
            user_metadata = user_result.document.metadata
            archetype = user_metadata.get("archetype", "")

            # Map archetypes to preferred restaurants
            if "foodie_explorer" in archetype:
                restaurant_preferences.extend(
                    ["mock_place_2", "mock_place_3", "mock_place_7"])  # Japanese, Thai, French
            elif "budget_conscious" in archetype:
                restaurant_preferences.extend(["mock_place_4", "mock_place_6"])  # American, Mexican
            elif "health_focused" in archetype:
                restaurant_preferences.extend(["mock_place_5", "mock_place_8"])  # Mediterranean, Vegetarian
            elif "business_diner" in archetype:
                restaurant_preferences.extend(["mock_place_9", "mock_place_7"])  # Steakhouse, French
            elif "comfort_seeker" in archetype:
                restaurant_preferences.extend(["mock_place_0", "mock_place_4"])  # Italian, American

        # Remove duplicates and limit
        unique_restaurants = list(set(restaurant_preferences))
        return unique_restaurants[:limit]

    async def find_restaurants_by_cuisine_embedding(self, cuisines: List[RestaurantCategory], limit: int = 10) -> List[
        VectorSearchResult]:
        """Find restaurants that match cuisine preferences using embeddings"""

        if not cuisines:
            return []

        # Create a query embedding representing the cuisine preferences
        query_embedding = np.zeros(1536)

        cuisine_map = {
            RestaurantCategory.ITALIAN: 0,
            RestaurantCategory.JAPANESE: 1,
            RestaurantCategory.THAI: 2,
            RestaurantCategory.INDIAN: 3,
            RestaurantCategory.AMERICAN: 4,
            RestaurantCategory.MEXICAN: 5,
            RestaurantCategory.FRENCH: 6,
            RestaurantCategory.CHINESE: 7,
            RestaurantCategory.MEDITERRANEAN: 8,
            RestaurantCategory.VEGETARIAN: 9
        }

        # Boost query embedding for each preferred cuisine
        for cuisine in cuisines:
            cuisine_idx = cuisine_map.get(cuisine, 0)
            start_idx = (cuisine_idx * 150) % 1536
            end_idx = min(start_idx + 100, 1536)
            query_embedding[start_idx:end_idx] += 0.8

        # Normalize
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search for matching restaurants
        results = await self.search_similar(
            query_embedding=query_embedding.tolist(),
            limit=limit,
            collection_name="restaurant_features"
        )

        return results