"""
Persistent conversation memory using SQLite/PostgreSQL.
Survives server restarts and enforces maximum context window.
"""
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.db_models import Message
from app.models.schemas import ChatMessage
from app.core.database import get_db_session
import config

logger = logging.getLogger(__name__)


class PersistentConversationMemory:
    """
    Manages persistent conversation memory using database storage.
    
    Stores messages in database with automatic context window enforcement.
    Designed to complement short-term memory as a cache layer.
    """
    
    def __init__(self, max_history: int = None):
        """
        Initialize persistent conversation memory.
        
        Args:
            max_history: Maximum number of messages to retrieve per session.
                        Defaults to config.settings.max_conversation_history
        """
        self.max_history = max_history or config.settings.max_conversation_history
        logger.info(f"PersistentConversationMemory initialized with max_history={self.max_history}")
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        db: Optional[Session] = None
    ) -> None:
        """
        Save a message to persistent storage.
        
        Args:
            session_id: Unique session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            db: Optional database session (creates new if not provided)
        """
        should_close = False
        if db is None:
            db = get_db_session()
            should_close = True
        
        try:
            message = Message(
                session_id=session_id,
                role=role,
                content=content
            )
            db.add(message)
            db.commit()
            logger.debug(f"Saved message to persistent storage: session_id={session_id}, role={role}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving message to database: {e}", exc_info=True)
            raise
        finally:
            if should_close:
                db.close()
    
    def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        db: Optional[Session] = None
    ) -> List[ChatMessage]:
        """
        Retrieve recent messages for a session from persistent storage.
        
        Returns messages in chronological order (oldest first).
        Automatically enforces maximum context window.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of messages to retrieve (defaults to max_history)
            db: Optional database session (creates new if not provided)
            
        Returns:
            List of ChatMessage objects in chronological order
        """
        should_close = False
        if db is None:
            db = get_db_session()
            should_close = True
        
        try:
            limit = limit or self.max_history
            
            # Query messages ordered by timestamp (oldest first)
            messages = db.query(Message)\
                .filter(Message.session_id == session_id)\
                .order_by(Message.timestamp.asc())\
                .limit(limit)\
                .all()
            
            # Convert to ChatMessage schema
            chat_messages = [
                ChatMessage(role=msg.role, content=msg.content)
                for msg in messages
            ]
            
            logger.debug(
                f"Retrieved {len(chat_messages)} messages from persistent storage "
                f"for session_id={session_id}"
            )
            
            return chat_messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages from database: {e}", exc_info=True)
            return []
        finally:
            if should_close:
                db.close()
    
    def get_message_count(self, session_id: str, db: Optional[Session] = None) -> int:
        """
        Get the total number of messages stored for a session.
        
        Args:
            session_id: Unique session identifier
            db: Optional database session (creates new if not provided)
            
        Returns:
            Total number of messages in the session
        """
        should_close = False
        if db is None:
            db = get_db_session()
            should_close = True
        
        try:
            count = db.query(Message)\
                .filter(Message.session_id == session_id)\
                .count()
            return count
        except Exception as e:
            logger.error(f"Error counting messages: {e}", exc_info=True)
            return 0
        finally:
            if should_close:
                db.close()
    
    def has_session(self, session_id: str, db: Optional[Session] = None) -> bool:
        """
        Check if a session has any messages in persistent storage.
        
        Args:
            session_id: Unique session identifier
            db: Optional database session (creates new if not provided)
            
        Returns:
            True if session exists, False otherwise
        """
        return self.get_message_count(session_id, db) > 0
    
    def clear_session(self, session_id: str, db: Optional[Session] = None) -> None:
        """
        Clear all messages for a session from persistent storage.
        
        Args:
            session_id: Unique session identifier
            db: Optional database session (creates new if not provided)
        """
        should_close = False
        if db is None:
            db = get_db_session()
            should_close = True
        
        try:
            deleted = db.query(Message)\
                .filter(Message.session_id == session_id)\
                .delete()
            db.commit()
            logger.info(f"Cleared {deleted} messages for session {session_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error clearing session: {e}", exc_info=True)
            raise
        finally:
            if should_close:
                db.close()


# Global persistent memory instance
persistent_memory = PersistentConversationMemory()
