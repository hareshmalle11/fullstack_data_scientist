from langchain_community.document_loaders import PyPDFLoader
import logging

logger = logging.getLogger(__name__)

def load_pdf(file_path: str):
    """
    Loads a PDF file using PyPDFLoader and returns a list of LangChain Documents.
    
    Args:
        file_path (str): Absolute or relative path to the PDF file.
        
    Returns:
        list: List of Document objects containing text content and page metadata.
    """
    try:
        logger.info(f"Starting to load PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} pages from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Failed to load PDF file at {file_path}: {str(e)}")
        raise RuntimeError(f"Error loading PDF {file_path}: {str(e)}")
