from .mock_adapter import VectorDBAdapter, VectorSearchResult, VectorDocument, MockVectorAdapter
# from .chroma_adapter import ChromaAdapter  # Comment out for now since it imports from base

__all__ = [
    "VectorDBAdapter",
    "VectorSearchResult",
    "VectorDocument",
    "MockVectorAdapter"
    # "ChromaAdapter"  # Comment out for now
]

# src/infrastructure/databases/vector_db/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding for storage/retrieval"""

    id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    content: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions"""
        return len(self.embedding)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""

    document: VectorDocument
    score: float  # Similarity score (higher = more similar)
    distance: float  # Distance metric (lower = more similar)

    @property
    def similarity_percentage(self) -> float:
        """Get similarity as percentage"""
        return min(self.score * 100, 100.0)


class VectorDBAdapter(ABC):
    """Abstract base class for vector database adapters"""

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector database"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the vector database"""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int,
                                metadata_schema: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection"""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections"""
        pass

    @abstractmethod
    async def insert_documents(self, documents: List[VectorDocument],
                               collection_name: Optional[str] = None) -> bool:
        """Insert multiple documents"""
        pass

    @abstractmethod
    async def insert_document(self, document: VectorDocument,
                              collection_name: Optional[str] = None) -> bool:
        """Insert a single document"""
        pass

    @abstractmethod
    async def search_similar(self, query_embedding: List[float],
                             limit: int = 10,
                             threshold: Optional[float] = None,
                             metadata_filter: Optional[Dict[str, Any]] = None,
                             collection_name: Optional[str] = None) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    async def get_document(self, document_id: str,
                           collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        """Get a document by ID"""
        pass

    @abstractmethod
    async def update_document(self, document: VectorDocument,
                              collection_name: Optional[str] = None) -> bool:
        """Update an existing document"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str,
                              collection_name: Optional[str] = None) -> bool:
        """Delete a document by ID"""
        pass

    @abstractmethod
    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """Count documents in collection"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            collections = await self.list_collections()
            return {
                "status": "healthy",
                "connected": self.is_connected,
                "collections": len(collections),
                "collection_names": collections
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }


# src/infrastructure/databases/vector_db/chroma_adapter.py

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import logging
import asyncio

from .base import VectorDBAdapter, VectorDocument, VectorSearchResult
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


class ChromaAdapter(VectorDBAdapter):
    """ChromaDB adapter for vector storage and similarity search"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 8000,
                 collection_name: str = "restaurant_recommendations"):

        super().__init__(collection_name)
        self.host = host
        self.port = port
        self.client = None
        self.collections = {}
        self.settings = get_settings()

    async def connect(self) -> bool:
        """Connect to ChromaDB"""
        try:
            # Create ChromaDB client
            chroma_settings = ChromaSettings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=self.host,
                chroma_server_http_port=self.port
            )

            self.client = chromadb.Client(chroma_settings)

            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.heartbeat
            )

            self.is_connected = True
            logger.info(f"Connected to ChromaDB at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from ChromaDB"""
        self.client = None
        self.collections = {}
        self.is_connected = False
        logger.info("Disconnected from ChromaDB")

    async def create_collection(self, collection_name: str, dimension: int,
                                metadata_schema: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection in ChromaDB"""
        try:
            if not self.client:
                raise RuntimeError("Not connected to ChromaDB")

            # Create collection with embedding function
            collection = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.create_collection(
                    name=collection_name,
                    metadata={"dimension": dimension, **(metadata_schema or {})}
                )
            )

            self.collections[collection_name] = collection
            logger.info(f"Created ChromaDB collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if not self.client:
                raise RuntimeError("Not connected to ChromaDB")

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.delete_collection(name=collection_name)
            )

            if collection_name in self.collections:
                del self.collections[collection_name]

            logger.info(f"Deleted ChromaDB collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            if not self.client:
                raise RuntimeError("Not connected to ChromaDB")

            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.client.list_collections
            )

            return [col.name for col in collections]

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def _get_collection(self, collection_name: str):
        """Get or create collection reference"""
        if collection_name in self.collections:
            return self.collections[collection_name]

        try:
            collection = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_collection(name=collection_name)
            )
            self.collections[collection_name] = collection
            return collection

        except Exception:
            # Collection doesn't exist, create it
            await self.create_collection(collection_name, 1536)  # Default dimension
            return self.collections[collection_name]

    async def insert_documents(self, documents: List[VectorDocument],
                               collection_name: Optional[str] = None) -> bool:
        """Insert multiple documents"""
        try:
            target_collection = collection_name or self.collection_name
            collection = await self._get_collection(target_collection)

            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            documents_content = [doc.content or "" for doc in documents]

            # Insert into ChromaDB
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_content
                )
            )

            logger.debug(f"Inserted {len(documents)} documents into {target_collection}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False

    async def insert_document(self, document: VectorDocument,
                              collection_name: Optional[str] = None) -> bool:
        """Insert a single document"""
        return await self.insert_documents([document], collection_name)

    async def search_similar(self, query_embedding: List[float],
                             limit: int = 10,
                             threshold: Optional[float] = None,
                             metadata_filter: Optional[Dict[str, Any]] = None,
                             collection_name: Optional[str] = None) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        try:
            target_collection = collection_name or self.collection_name
            collection = await self._get_collection(target_collection)

            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": limit
            }

            if metadata_filter:
                query_params["where"] = metadata_filter

            # Perform search
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.query(**query_params)
            )

            # Convert results to VectorSearchResult objects
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    embedding = results.get("embeddings", [[]])[0][i] if results.get("embeddings") else []
                    metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                    content = results.get("documents", [[]])[0][i] if results.get("documents") else None
                    distance = results.get("distances", [[]])[0][i] if results.get("distances") else 0.0

                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1.0 / (1.0 + distance) if distance > 0 else 1.0

                    # Apply threshold filter if specified
                    if threshold is not None and similarity_score < threshold:
                        continue

                    document = VectorDocument(
                        id=doc_id,
                        embedding=embedding,
                        metadata=metadata,
                        content=content
                    )

                    result = VectorSearchResult(
                        document=document,
                        score=similarity_score,
                        distance=distance
                    )

                    search_results.append(result)

            logger.debug(f"Found {len(search_results)} similar documents in {target_collection}")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            return []

    async def get_document(self, document_id: str,
                           collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        """Get a document by ID"""
        try:
            target_collection = collection_name or self.collection_name
            collection = await self._get_collection(target_collection)

            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.get(ids=[document_id], include=["embeddings", "metadatas", "documents"])
            )

            if results["ids"] and results["ids"][0]:
                embedding = results.get("embeddings", [[]])[0] if results.get("embeddings") else []
                metadata = results.get("metadatas", [[]])[0] if results.get("metadatas") else {}
                content = results.get("documents", [[]])[0] if results.get("documents") else None

                return VectorDocument(
                    id=document_id,
                    embedding=embedding,
                    metadata=metadata,
                    content=content
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def update_document(self, document: VectorDocument,
                              collection_name: Optional[str] = None) -> bool:
        """Update an existing document"""
        try:
            target_collection = collection_name or self.collection_name
            collection = await self._get_collection(target_collection)

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.update(
                    ids=[document.id],
                    embeddings=[document.embedding],
                    metadatas=[document.metadata],
                    documents=[document.content or ""]
                )
            )

            logger.debug(f"Updated document {document.id} in {target_collection}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            return False

    async def delete_document(self, document_id: str,
                              collection_name: Optional[str] = None) -> bool:
        """Delete a document by ID"""
        try:
            target_collection = collection_name or self.collection_name
            collection = await self._get_collection(target_collection)

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.delete(ids=[document_id])
            )

            logger.debug(f"Deleted document {document_id} from {target_collection}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """Count documents in collection"""
        try:
            target_collection = collection_name or self.collection_name
            collection = await self._get_collection(target_collection)

            result = await asyncio.get_event_loop().run_in_executor(
                None, collection.count
            )

            return result

        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0