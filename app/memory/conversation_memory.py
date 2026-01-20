"""
Short-term conversation memory management.
In-memory storage for conversation history with automatic trimming.
"""
import logging
from typing import List, Dict
from app.models.schemas import ChatMessage
import config

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages short-term conversation memory in-memory.
    
    Stores conversation history per session with automatic trimming
    when max_history limit is exceeded. Designed to be easily
    replaceable with persistent storage or vector memory later.
    """
    
    def __init__(self, max_history: int = None):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep per session.
                        Defaults to config.settings.max_conversation_history
        """
        self.max_history = max_history or config.settings.max_conversation_history
        self._storage: Dict[str, List[ChatMessage]] = {}
        logger.info(f"ConversationMemory initialized with max_history={self.max_history}")
    
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            List of ChatMessage objects for the session (empty list if new session)
        """
        return self._storage.get(session_id, []).copy()
    
    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """
        Add a message to the conversation history for a session.
        
        Automatically trims old messages if max_history is exceeded.
        Maintains chronological order (oldest first).
        
        Args:
            session_id: Unique session identifier
            message: ChatMessage to add
        """
        if session_id not in self._storage:
            self._storage[session_id] = []
        
        self._storage[session_id].append(message)
        
        # Trim if exceeds max_history
        if len(self._storage[session_id]) > self.max_history:
            # Keep only the most recent messages
            self._storage[session_id] = self._storage[session_id][-self.max_history:]
            logger.debug(
                f"Trimmed conversation history for session {session_id} "
                f"to {self.max_history} messages"
            )
    
    def clear(self, session_id: str) -> None:
        """
        Clear all conversation history for a session.
        
        Args:
            session_id: Unique session identifier
        """
        if session_id in self._storage:
            del self._storage[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
        else:
            logger.debug(f"Attempted to clear non-existent session {session_id}")
    
    def get_message_count(self, session_id: str) -> int:
        """
        Get the number of messages stored for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Number of messages in the session
        """
        return len(self._storage.get(session_id, []))
    
    def has_session(self, session_id: str) -> bool:
        """
        Check if a session exists in memory.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        return session_id in self._storage


# Global memory instance
conversation_memory = ConversationMemory()
