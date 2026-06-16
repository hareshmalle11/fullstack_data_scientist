import ollama
from voice_assistant.config.settings import Settings
from voice_assistant.utils.logger import logger
from voice_assistant.utils.exceptions import OllamaNotRunningException

def check_connection() -> bool:
    """Verifies that the local Ollama service is running and accessible."""
    try:
        # Simple client check
        ollama.list()
        return True
    except Exception as e:
        logger.error(f"Failed connection check to Ollama: {e}")
        return False

def check_model_exists(model_name: str = Settings.MODEL_NAME) -> bool:
    """Checks if the requested model exists in local Ollama storage."""
    try:
        models_list = ollama.list()
        models = models_list.get("models", [])
        for model in models:
            if model.get("name") == model_name or model.get("model") == model_name:
                return True
        return False
    except Exception as e:
        logger.error(f"Failed to fetch model list from Ollama: {e}")
        return False

def generate_response(messages: list) -> str:
    """Sends a chat message payload to Ollama and returns the textual response.
    
    Args:
        messages: A list of dicts structured as:
                  [{'role': 'system'/'user'/'assistant', 'content': '...'}]
                  
    Returns:
        The generated assistant response string.
    """
    model_name = Settings.MODEL_NAME
    logger.info(f"Sending request to Ollama LLM ({model_name}). Payload message count: {len(messages)}")
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": 0.3,  # Lower temperature is better for concise/accurate voice replies
                "top_p": 0.9,
            }
        )
        assistant_text = response.get("message", {}).get("content", "").strip()
        logger.info("Received response from Ollama.")
        return assistant_text
    except Exception as e:
        logger.error(f"Ollama generation request failed: {e}")
        # Check if Ollama is unresponsive
        if not check_connection():
            raise OllamaNotRunningException("Ollama daemon is unresponsive or not running.") from e
        raise e
