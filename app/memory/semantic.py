"""
Semantic memory management using embeddings and vector search.
Stores and retrieves user-specific facts and preferences.
"""
import logging
from typing import List, Optional, Dict
from app.services.embedding_service import embedding_service
from app.memory.vector_store import vector_store, MemoryEntry
from app.memory.memory_extractor import memory_extractor
import config

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Manages semantic long-term memory using embeddings.
    
    Stores meaningful facts and preferences, retrieves them semantically
    based on query meaning rather than exact text matching.
    """
    
    def __init__(self):
        """Initialize semantic memory."""
        self.min_similarity = config.settings.semantic_memory_min_similarity
        self.max_retrieved = config.settings.semantic_memory_max_retrieved
        logger.info("SemanticMemory initialized")
    
    async def store_memory(
        self,
        text: str,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a memory with semantic embedding.
        
        Args:
            text: Memory text to store
            user_id: User identifier
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        try:
            # Generate embedding
            embedding = await embedding_service.embed(text)
            
            # Store in vector store
            memory_id = vector_store.store(
                text=text,
                embedding=embedding,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            logger.info(f"Stored semantic memory {memory_id} for user {user_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing semantic memory: {e}", exc_info=True)
            raise
    
    async def retrieve_relevant(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve semantically relevant memories for a query.
        
        Args:
            query: Search query text
            user_id: User identifier
            top_k: Number of memories to retrieve (defaults to config)
            
        Returns:
            List of MemoryEntry objects, sorted by relevance
        """
        try:
            # Generate query embedding
            query_embedding = await embedding_service.embed(query)
            
            # Search vector store
            top_k = top_k or self.max_retrieved
            results = vector_store.search(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=top_k,
                min_similarity=self.min_similarity
            )
            
            # Extract MemoryEntry objects (results are tuples of (entry, similarity))
            memories = [entry for entry, _ in results]
            
            logger.debug(
                f"Retrieved {len(memories)} relevant memories for user {user_id} "
                f"with query: {query[:50]}..."
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving semantic memories: {e}", exc_info=True)
            return []
    
    async def extract_and_store(
        self,
        messages: List,
        user_id: str
    ) -> List[str]:
        """
        Extract candidate memories from conversation and store them.
        
        Args:
            messages: List of ChatMessage objects
            user_id: User identifier
            
        Returns:
            List of stored memory IDs
        """
        stored_ids = []
        
        # Extract candidate memories
        candidates = memory_extractor.extract_candidates(messages)
        
        for candidate in candidates:
            if memory_extractor.should_store(candidate):
                try:
                    memory_id = await self.store_memory(
                        text=candidate["text"],
                        user_id=user_id,
                        metadata={
                            **(candidate.get("metadata", {})),
                            "confidence": candidate.get("confidence", 0),
                            "source": candidate.get("source_message", "")[:100]
                        }
                    )
                    stored_ids.append(memory_id)
                except Exception as e:
                    logger.warning(f"Failed to store memory candidate: {e}")
        
        if stored_ids:
            logger.info(f"Extracted and stored {len(stored_ids)} memories for user {user_id}")
        
        return stored_ids
    
    def format_memories_for_prompt(self, memories: List[MemoryEntry]) -> str:
        """
        Format memories into a string for injection into system prompt.
        
        Args:
            memories: List of MemoryEntry objects
            
        Returns:
            Formatted string with memories
        """
        if not memories:
            return ""
        
        formatted = "\n\n## User Context & Preferences:\n"
        
        for i, memory in enumerate(memories, 1):
            formatted += f"{i}. {memory.text}\n"
        
        formatted += "\nUse this information to provide personalized responses.\n"
        
        return formatted
    
    async def get_context_for_query(
        self,
        query: str,
        user_id: str
    ) -> str:
        """
        Get formatted context string with relevant memories for a query.
        
        Args:
            query: User query text
            user_id: User identifier
            
        Returns:
            Formatted context string (empty if no memories found)
        """
        memories = await self.retrieve_relevant(query, user_id)
        return self.format_memories_for_prompt(memories)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        return vector_store.delete(memory_id)
    
    def get_user_memories(self, user_id: str) -> List[MemoryEntry]:
        """Get all memories for a user."""
        return vector_store.get_user_memories(user_id)
    
    def clear_user_memories(self, user_id: str) -> None:
        """Clear all memories for a user."""
        vector_store.clear_user(user_id)
        logger.info(f"Cleared all semantic memories for user {user_id}")


# Global semantic memory instance
semantic_memory = SemanticMemory()
