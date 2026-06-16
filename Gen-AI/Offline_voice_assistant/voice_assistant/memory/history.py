from datetime import datetime
from voice_assistant.memory.database import get_connection
from voice_assistant.utils.logger import logger

def save_message(session_id: str, role: str, message: str) -> bool:
    """Saves a message to the SQLite chat history.
    
    Args:
        session_id: Unique identifier for the conversation session
        role: Message speaker role ('user' or 'assistant')
        message: Content of the message
    """
    timestamp = datetime.now().isoformat()
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, timestamp, role, message) VALUES (?, ?, ?, ?)",
            (session_id, timestamp, role, message)
        )
        conn.commit()
        logger.info(f"Saved message from {role} to SQLite for session {session_id}.")
        return True
    except Exception as e:
        logger.error(f"Failed to save message to SQLite: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_recent_messages(session_id: str, limit: int = 20) -> list:
    """Retrieves recent conversation messages for a specific session ID.
    
    Args:
        session_id: Unique identifier for the conversation session
        limit: Max number of messages to fetch
        
    Returns:
        List of dictionaries with role and message content.
    """
    conn = None
    messages = []
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, message FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        for row in rows:
            messages.append({
                "role": row["role"],
                "content": row["message"]
            })
        logger.info(f"Retrieved {len(messages)} past messages from database for session {session_id}.")
    except Exception as e:
        logger.error(f"Failed to retrieve chat history from SQLite: {e}")
    finally:
        if conn:
            conn.close()
            
    return messages
