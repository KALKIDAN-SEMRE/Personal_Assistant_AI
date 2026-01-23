"""
Simple in-memory vector store for semantic memory.
Can be replaced with FAISS, Chroma, or managed vector DB later.
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry with metadata."""
    id: str
    text: str
    embedding: List[float]
    user_id: str
    metadata: Dict
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class InMemoryVectorStore:
    """
    Simple in-memory vector store for semantic search.
    Uses cosine similarity for retrieval.
    Designed to be easily replaceable with FAISS/Chroma/Pinecone.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize vector store.
        
        Args:
            max_size: Maximum number of memories to store
        """
        self._memories: Dict[str, MemoryEntry] = {}
        self._user_memories: Dict[str, List[str]] = {}  # user_id -> [memory_ids]
        self.max_size = max_size
        self._next_id = 0
        logger.info(f"InMemoryVectorStore initialized with max_size={max_size}")
    
    def store(
        self,
        text: str,
        embedding: List[float],
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a memory with its embedding.
        
        Args:
            text: Memory text content
            embedding: Embedding vector
            user_id: User identifier
            metadata: Optional metadata dictionary
            
        Returns:
            Memory ID
        """
        memory_id = str(self._next_id)
        self._next_id += 1
        
        entry = MemoryEntry(
            id=memory_id,
            text=text,
            embedding=embedding,
            user_id=user_id,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        self._memories[memory_id] = entry
        
        # Track user memories
        if user_id not in self._user_memories:
            self._user_memories[user_id] = []
        self._user_memories[user_id].append(memory_id)
        
        # Enforce max size (remove oldest)
        if len(self._memories) > self.max_size:
            self._evict_oldest()
        
        logger.debug(f"Stored memory {memory_id} for user {user_id}")
        return memory_id
    
    def search(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar memories using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            user_id: User identifier (only search this user's memories)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples, sorted by similarity
        """
        if user_id not in self._user_memories:
            return []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        results = []
        
        # Only search user's memories
        for memory_id in self._user_memories[user_id]:
            entry = self._memories.get(memory_id)
            if not entry:
                continue
            
            # Calculate cosine similarity
            memory_vec = np.array(entry.embedding)
            memory_norm = np.linalg.norm(memory_vec)
            
            if memory_norm == 0:
                continue
            
            similarity = np.dot(query_vec, memory_vec) / (query_norm * memory_norm)
            
            if similarity >= min_similarity:
                results.append((entry, float(similarity)))
                # Update access tracking
                entry.access_count += 1
                entry.last_accessed = datetime.now()
        
        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        return self._memories.get(memory_id)
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if memory_id not in self._memories:
            return False
        
        entry = self._memories[memory_id]
        user_id = entry.user_id
        
        # Remove from user's memory list
        if user_id in self._user_memories:
            self._user_memories[user_id] = [
                mid for mid in self._user_memories[user_id] if mid != memory_id
            ]
        
        del self._memories[memory_id]
        logger.debug(f"Deleted memory {memory_id}")
        return True
    
    def get_user_memories(self, user_id: str) -> List[MemoryEntry]:
        """Get all memories for a user."""
        if user_id not in self._user_memories:
            return []
        
        return [
            self._memories[mid]
            for mid in self._user_memories[user_id]
            if mid in self._memories
        ]
    
    def _evict_oldest(self) -> None:
        """Remove the oldest memory when max size is exceeded."""
        if not self._memories:
            return
        
        # Find oldest memory (by created_at, then by access_count)
        oldest = min(
            self._memories.values(),
            key=lambda e: (e.created_at, -e.access_count)
        )
        
        self.delete(oldest.id)
        logger.debug(f"Evicted oldest memory {oldest.id} due to max size limit")
    
    def clear_user(self, user_id: str) -> None:
        """Clear all memories for a user."""
        if user_id not in self._user_memories:
            return
        
        memory_ids = list(self._user_memories[user_id])
        for memory_id in memory_ids:
            del self._memories[memory_id]
        
        del self._user_memories[user_id]
        logger.info(f"Cleared all memories for user {user_id}")


# Global vector store instance
vector_store = InMemoryVectorStore()
