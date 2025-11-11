"""
Database connection and session management
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from src.core.config import settings

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Global engine instance
engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[sessionmaker] = None


async def init_db():
    """Initialize database connection"""
    global engine, AsyncSessionLocal

    try:
        # Create async engine
        engine = create_async_engine(
            settings.database_url,
            echo=settings.DEBUG,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_pre_ping=True,
            future=True
        )

        # Create session factory
        AsyncSessionLocal = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Test connection and create schema if not exists
        async with engine.begin() as conn:
            # Create schema if not exists
            await conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {settings.DB_SCHEMA}")
            )
            await conn.commit()

        logger.info("Database connection initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db():
    """Close database connection"""
    global engine

    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


async def check_db_health() -> bool:
    """Check database health"""
    if not engine:
        return False

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for FastAPI dependency injection"""
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized")

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for manual context management"""
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized")

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def execute_query(query: str, params: dict = None):
    """Execute a raw SQL query"""
    async with get_db_context() as db:
        result = await db.execute(text(query), params or {})
        await db.commit()
        return result


async def fetch_one(query: str, params: dict = None):
    """Fetch a single row"""
    async with get_db_context() as db:
        result = await db.execute(text(query), params or {})
        return result.fetchone()


async def fetch_all(query: str, params: dict = None):
    """Fetch all rows"""
    async with get_db_context() as db:
        result = await db.execute(text(query), params or {})
        return result.fetchall()