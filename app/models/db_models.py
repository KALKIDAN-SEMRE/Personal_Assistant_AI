"""
Database models for persistent storage.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.sql import func
from app.core.database import Base


class Message(Base):
    """
    Message model for storing conversation history.
    
    Stores individual messages (user/assistant) with session tracking
    and timestamps for chronological ordering.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Index for efficient queries by session_id and timestamp
    __table_args__ = (
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, session_id={self.session_id}, role={self.role})>"
