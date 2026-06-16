from voice_assistant.utils.logger import logger

def truncate_context(history_messages: list, max_turns: int = 10) -> list:
    """Limits the number of context message history turns sent to the LLM.
    
    Args:
        history_messages: List of message dicts.
        max_turns: Maximum number of messages to keep in history context (default 10).
        
    Returns:
        A sublist containing only the most recent messages.
    """
    if len(history_messages) <= max_turns:
        return history_messages
        
    truncated = history_messages[-max_turns:]
    logger.info(f"Context truncated from {len(history_messages)} to {len(truncated)} messages.")
    return truncated
