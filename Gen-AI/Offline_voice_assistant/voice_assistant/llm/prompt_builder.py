from voice_assistant.config.settings import Settings
from voice_assistant.config.constants import DEFAULT_SYSTEM_PROMPT
from voice_assistant.utils.logger import logger

def load_system_prompt() -> str:
    """Loads system instructions from prompts/system_prompt.txt or returns defaults."""
    prompt_file = Settings.SYSTEM_PROMPT_FILE
    if prompt_file.exists():
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            logger.error(f"Failed to read system prompt file: {e}")
            
    logger.info("Using default hardcoded system prompt.")
    return DEFAULT_SYSTEM_PROMPT

def build_chat_messages(history_messages: list, current_user_message: str) -> list:
    """Builds the full chat messages array for the Ollama chat endpoint.
    
    Args:
        history_messages: List of past messages (dicts with 'role' and 'content')
        current_user_message: The newly transcribed user message.
        
    Returns:
        Structured message list containing system prompt, history, and user turn.
    """
    system_prompt = load_system_prompt()
    
    messages = []
    # Add System Prompt
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # Add Past Conversation Context
    for msg in history_messages:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
        
    # Add Current User Message
    messages.append({
        "role": "user",
        "content": current_user_message
    })
    
    return messages
