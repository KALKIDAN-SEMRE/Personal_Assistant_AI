"""
Database connection and session management.
PostgreSQL-ready design using SQLAlchemy.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import config

logger = logging.getLogger(__name__)

# Create database engine (PostgreSQL-ready)
engine = create_engine(
    config.settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in config.settings.database_url else {},
    echo=config.settings.debug  # Log SQL queries in debug mode
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def init_db() -> None:
    """
    Initialize database - create all tables.
    Should be called on application startup.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized and tables created")


@contextmanager
def get_db() -> Session:
    """
    Context manager for database sessions.
    Ensures proper cleanup after use.
    
    Usage:
        with get_db() as db:
            # use db session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session (for dependency injection).
    Use get_db() context manager when possible.
    
    Returns:
        Database session
    """
    return SessionLocal()
