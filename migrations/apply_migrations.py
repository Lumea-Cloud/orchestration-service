#!/usr/bin/env python3
"""
Apply database migrations for Orchestration Service
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add parent directory to path so we can import from src
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_applied_migrations(engine) -> List[str]:
    """Get list of already applied migrations"""
    try:
        async with engine.begin() as conn:
            # Create migrations tracking table if it doesn't exist
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {settings.DB_SCHEMA}.schema_migrations (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Get list of applied migrations
            result = await conn.execute(text(f"""
                SELECT filename FROM {settings.DB_SCHEMA}.schema_migrations
                ORDER BY filename
            """))

            rows = result.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"Error getting applied migrations: {e}")
        return []


async def apply_migration(engine, migration_file: Path) -> bool:
    """Apply a single migration file"""
    try:
        logger.info(f"Applying migration: {migration_file.name}")

        # Read migration SQL
        with open(migration_file, 'r') as f:
            sql = f.read()

        # Execute migration in a transaction
        async with engine.begin() as conn:
            # Split SQL into statements and execute each
            statements = [s.strip() for s in sql.split(';') if s.strip()]
            for statement in statements:
                await conn.execute(text(statement))

            # Record migration as applied
            await conn.execute(text(f"""
                INSERT INTO {settings.DB_SCHEMA}.schema_migrations (filename)
                VALUES (:filename)
                ON CONFLICT (filename) DO NOTHING
            """), {"filename": migration_file.name})

        logger.info(f"Successfully applied migration: {migration_file.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to apply migration {migration_file.name}: {e}")
        return False


async def apply_all_migrations():
    """Apply all pending migrations"""
    # Get migrations directory
    migrations_dir = Path(__file__).parent

    # Create async engine
    DATABASE_URL = (
        f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )

    engine = create_async_engine(DATABASE_URL, echo=False)

    try:
        # Ensure schema exists
        async with engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.DB_SCHEMA}"))

        # Get list of applied migrations
        applied = await get_applied_migrations(engine)
        logger.info(f"Already applied migrations: {applied}")

        # Get list of migration files
        migration_files = sorted([
            f for f in migrations_dir.glob("*.sql")
            if f.is_file()
        ])

        # Apply pending migrations
        pending_count = 0
        for migration_file in migration_files:
            if migration_file.name not in applied:
                pending_count += 1
                success = await apply_migration(engine, migration_file)
                if not success:
                    logger.error("Migration failed, stopping")
                    break

        if pending_count == 0:
            logger.info("No pending migrations to apply")
        else:
            logger.info(f"Applied {pending_count} migration(s)")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(apply_all_migrations())