import time
from voice_assistant.llm.prompt_builder import build_chat_messages
from voice_assistant.llm.context_manager import truncate_context
from voice_assistant.llm.ollama_client import generate_response
from voice_assistant.utils.logger import logger

class ChatService:
    def get_assistant_reply(self, user_message: str, history: list) -> tuple[str, float]:
        """Constructs prompt, sends it to Ollama, and returns response and response time.
        
        Args:
            user_message: Newly transcribed user text
            history: List of past messages in the conversation
            
        Returns:
            Tuple of (response_text, elapsed_time_seconds)
        """
        logger.info("ChatService: Preparing prompt payload...")
        
        # Truncate conversation history to avoid overflowing model context window
        truncated_history = truncate_context(history, max_turns=10)
        
        # Build structured message payload
        messages = build_chat_messages(truncated_history, user_message)
        
        # Record LLM generation time
        start_time = time.time()
        reply_text = generate_response(messages)
        elapsed_time = time.time() - start_time
        
        logger.info(f"ChatService: Generated response in {elapsed_time:.2f} seconds.")
        return reply_text, elapsed_time
