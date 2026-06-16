from langchain_community.embeddings import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)

def get_embeddings():
    """
    Initializes and returns OllamaEmbeddings configured for local nomic-embed-text.
    
    Returns:
        OllamaEmbeddings: LangChain compatible embedding generator.
    """
    try:
        logger.info("Initializing OllamaEmbeddings using model: nomic-embed-text")
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OllamaEmbeddings: {str(e)}")
        raise RuntimeError(f"Error initializing OllamaEmbeddings: {str(e)}")
