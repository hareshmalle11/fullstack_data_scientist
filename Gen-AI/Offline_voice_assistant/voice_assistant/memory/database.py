import sqlite3
import os
from voice_assistant.config.settings import Settings
from voice_assistant.config.constants import DB_MESSAGES_TABLE_SCHEMA
from voice_assistant.utils.logger import logger
from voice_assistant.utils.exceptions import DatabaseLockedException

def get_connection():
    """Establishes and returns a connection to the SQLite database."""
    db_path = Settings.DB_PATH
    
    # Ensure database folder exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        # Return rows as dictionaries for clean access
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite connection error: {e}")
        if "locked" in str(e).lower():
            raise DatabaseLockedException("Database is currently locked by another process.") from e
        raise e

def create_database():
    """Initializes the database schema if tables do not exist."""
    logger.info("Initializing SQLite database and verifying tables...")
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(DB_MESSAGES_TABLE_SCHEMA)
        conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        raise e
    finally:
        if conn:
            conn.close()
