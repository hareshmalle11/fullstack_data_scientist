import re
import logging
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def split_into_sentences(text: str):
    """Splits a block of text into sentences using a robust regex pattern."""
    # Split by periods, question marks, or exclamation marks followed by whitespace,
    # avoiding common titles like Mr, Ms, Dr, etc.
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def semantic_chunk_text(text: str, embeddings_model, similarity_threshold=0.65, min_chunk_len=300, max_chunk_len=1500):
    """
    Splits text into chunks semantically based on sentence embeddings and cosine similarities.
    
    Args:
        text (str): The page or section text.
        embeddings_model: The LangChain embeddings instance.
        similarity_threshold (float): Threshold below which a new chunk is started.
        min_chunk_len (int): Minimum characters before we allow a split.
        max_chunk_len (int): Maximum characters after which we force a split.
        
    Returns:
        list of str: The semantic text chunks.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 2:
        return [text] if text.strip() else []

    try:
        # Batch embed all sentences to optimize performance
        embeddings = embeddings_model.embed_documents(sentences)
    except Exception as e:
        logger.warning(f"Failed to generate sentence embeddings for semantic chunking: {e}. Falling back.")
        return None  # Let the caller fall back to recursive splitter

    chunks = []
    current_chunk = []
    current_len = 0

    for i in range(len(sentences)):
        current_chunk.append(sentences[i])
        current_len += len(sentences[i]) + 1

        # Check if we should split before adding the next sentence
        if i < len(sentences) - 1:
            similarity = cosine_similarity(embeddings[i], embeddings[i+1])
            
            # Split conditions:
            # 1. Similarity is below threshold and we have reached min length
            # 2. Or the current chunk is getting too large
            if (similarity < similarity_threshold and current_len >= min_chunk_len) or (current_len >= max_chunk_len):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

    # Append any remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_document(docs, doc_id: str, filename: str, embeddings_model, use_semantic=True, similarity_threshold=0.65, chunk_size=1000, chunk_overlap=200):
    """
    Chunks a list of page documents and returns chunk documents with metadata.
    
    Args:
        docs (list of Document): Pages loaded from PyPDFLoader.
        doc_id (str): Unique document identifier.
        filename (str): Name of the source file.
        embeddings_model: Embeddings generator to use for semantic chunking.
        use_semantic (bool): Whether to attempt semantic chunking first.
        similarity_threshold (float): Cosine similarity threshold for semantic breaks.
        chunk_size (int): Recursive splitter chunk size (fallback).
        chunk_overlap (int): Recursive splitter overlap (fallback).
        
    Returns:
        list of Document: Chunked documents with filled metadata.
    """
    logger.info(f"Starting chunking for document: {filename} (ID: {doc_id}) using semantic={use_semantic}")
    
    chunk_docs = []
    chunk_count = 0
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1  # 1-indexed page
        page_text = doc.page_content.strip()
        
        if not page_text:
            continue
            
        chunks = None
        
        # Try semantic chunking first if requested
        if use_semantic and embeddings_model is not None:
            chunks = semantic_chunk_text(
                page_text, 
                embeddings_model, 
                similarity_threshold=similarity_threshold
            )
            
        # Fallback to recursive character text splitting if semantic chunking is disabled or failed
        if chunks is None:
            logger.debug(f"Using RecursiveCharacterTextSplitter for {filename} page {page_num}")
            chunks = recursive_splitter.split_text(page_text)
            
        # Wrap each chunk text in a Document with metadata
        for text in chunks:
            if not text.strip():
                continue
                
            metadata = {
                "document_id": doc_id,
                "filename": filename,
                "page": page_num,
                "chunk_index": chunk_count
            }
            chunk_docs.append(Document(page_content=text, metadata=metadata))
            chunk_count += 1
            
    logger.info(f"Completed chunking for {filename}: created {len(chunk_docs)} chunks.")
    return chunk_docs
