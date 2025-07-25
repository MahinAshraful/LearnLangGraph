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