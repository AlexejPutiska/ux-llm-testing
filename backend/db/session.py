"""
Database session configuration.

Creates the SQLAlchemy engine and session factory used throughout the application.
The DATABASE_URL can be overridden via the DATABASE_URL environment variable;
if not set, falls back to the local development default.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use DATABASE_URL from environment if provided, otherwise fall back to local default.
# In production or Docker environments, set DATABASE_URL as an environment variable.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:aaa123@127.0.0.1:5432/bp",
)

# pool_pre_ping=True verifies the connection is alive before each use,
# which prevents errors after database restarts or idle timeouts.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory: autocommit and autoflush are disabled so that transactions
# are explicitly committed by the caller.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
