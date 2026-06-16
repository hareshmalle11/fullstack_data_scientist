import os
from datetime import datetime
from pathlib import Path
from voice_assistant.config.settings import Settings
from voice_assistant.memory.history import save_message, get_recent_messages
from voice_assistant.utils.logger import logger

class MemoryService:
    def __init__(self, session_id: str = None):
        """Initializes memory cache, loading past history for the session.
        
        If no session ID is supplied, it attempts to load the most recent active session
        from sessions/current_session.txt, or generates a new timestamp-based session ID.
        """
        self.session_id = session_id or self._load_or_create_session_id()
        self.session_history = []
        self._load_history_to_ram()

    def _load_or_create_session_id(self) -> str:
        """Reads the current session ID from file, or creates a new one."""
        session_file = Settings.CURRENT_SESSION_FILE
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    sid = f.read().strip()
                    if sid:
                        logger.info(f"Loaded existing session ID: {sid}")
                        return sid
            except Exception as e:
                logger.error(f"Failed to read current session ID file: {e}")
                
        # Generate new session ID
        new_sid = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                f.write(new_sid)
            logger.info(f"Created and saved new session ID: {new_sid}")
        except Exception as e:
            logger.error(f"Failed to write current session ID file: {e}")
            
        return new_sid

    def _load_history_to_ram(self):
        """Queries SQLite and caches recent messages for the current session in RAM."""
        logger.info(f"Caching SQLite history to RAM for session: {self.session_id}")
        # Retrieve recent 20 messages (10 user/assistant turns)
        db_messages = get_recent_messages(self.session_id, limit=30)
        self.session_history = db_messages
        logger.info(f"In-RAM session history initialized with {len(self.session_history)} messages.")

    def add_message(self, role: str, content: str):
        """Appends a new turn to session memory and SQLite DB.
        
        Args:
            role: 'user' or 'assistant'
            content: The text content of the message
        """
        # Append to RAM cache (structured for prompt builder: {'role': ..., 'content': ...})
        self.session_history.append({
            "role": role,
            "content": content
        })
        
        # Save persistently to SQLite DB
        save_message(self.session_id, role, content)

    def get_history(self) -> list:
        """Returns the active cached list of conversation messages."""
        return self.session_history

    def start_new_session(self) -> str:
        """Forces the creation of a new session ID and clears the RAM cache."""
        new_sid = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            with open(Settings.CURRENT_SESSION_FILE, "w", encoding="utf-8") as f:
                f.write(new_sid)
            self.session_id = new_sid
            self.session_history = []
            logger.info(f"Force-started a new session: {new_sid}. RAM history cleared.")
        except Exception as e:
            logger.error(f"Failed to update current session file on new session request: {e}")
            
        return new_sid
