import logging
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from utils.embeddings import get_embeddings
from ingest import get_chroma_db

logger = logging.getLogger(__name__)

def list_ollama_models():
    """Lists the available LLM models on the local Ollama instance."""
    try:
        models_data = ollama.list()
        # Exclude embedding models from the text generation list
        models = [
            m['model'] for m in models_data.get('models', []) 
            if 'embed' not in m['model']
        ]
        return models
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return ["qwen2.5-coder:latest", "deepseek-r1:1.5b"]

def compute_similarity_percentage(distance: float) -> float:
    """
    Converts Chroma distance (typically L2 distance) to a similarity percentage.
    Lower L2 distance means higher similarity.
    """
    # For L2 distance in Chroma, 0.0 is exact match.
    # We map distance to a 0% - 100% similarity score
    # Usually L2 distance is in range [0, 2] for normalized embeddings.
    score = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, score)) * 100.0

def retrieve_context(query: str, k: int = 4):
    """
    Retrieves the most similar chunks from ChromaDB for a query.
    
    Returns:
        list of dict: Source chunks with metadata and similarity scores.
    """
    try:
        db = get_chroma_db()
        # Chroma similarity_search_with_score returns list of (Doc, distance)
        results = db.similarity_search_with_score(query, k=k)
        
        sources = []
        for doc, distance in results:
            sim_score = compute_similarity_percentage(distance)
            sources.append({
                "content": doc.page_content,
                "document_id": doc.metadata.get("document_id"),
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "similarity_score": sim_score
            })
        return sources
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

def format_rag_prompt(query: str, sources: list):
    """Formats the system instructions, context, and user query into a prompt."""
    context_str = ""
    for idx, src in enumerate(sources):
        context_str += f"Source [{idx+1}] - File: {src['filename']}, Page: {src['page']}\n"
        context_str += f"{src['content']}\n\n"
        
    prompt = f"""You are an expert AI Assistant working with a Retrieval-Augmented Generation (RAG) system.
Your job is to answer the user's question using only the retrieved contexts provided below.
Provide a direct, clear, detailed, and well-structured answer.

CRITICAL INSTRUCTION:
Do NOT mention 'Source [1]', 'Source [2]', page numbers, or phrases like 'retrieved contexts' or 'according to the text' in your answer. Provide a direct, factual answer based on the context without any citations, references, or meta-commentary about where the information is located. The user has buttons to inspect the sources separately, so your output must contain ONLY the answer.

If the retrieved context does not contain enough information to answer the question, state that clearly (e.g., "The provided documents do not contain enough information to answer this question."). Do not make up facts.

Retrieved Contexts:
---------------------
{context_str}
---------------------

User Question: {query}

Answer:"""
    return prompt

def query_rag(query: str, llm_model: str = "qwen2.5-coder:latest", k: int = 4):
    """
    Orchestrates the retrieval and generation phases.
    
    Returns:
        dict: containing 'answer' (str), 'sources' (list), and 'prompt' (str) used.
    """
    logger.info(f"Querying RAG system: '{query}' using LLM model: {llm_model}")
    
    # 1. Retrieve chunks
    sources = retrieve_context(query, k=k)
    
    if not sources:
        return {
            "answer": "No relevant documents found in the database. Please upload and index documents first.",
            "sources": [],
            "prompt": ""
        }
        
    # 2. Build prompt
    prompt = format_rag_prompt(query, sources)
    
    # 3. Query LLM
    try:
        llm = Ollama(
            model=llm_model,
            base_url="http://localhost:11434"
        )
        answer = llm.invoke(prompt)
        return {
            "answer": answer,
            "sources": sources,
            "prompt": prompt
        }
    except Exception as e:
        logger.error(f"Error querying Ollama LLM: {e}")
        return {
            "answer": f"Error communicating with local LLM server: {e}. Please ensure Ollama is running.",
            "sources": sources,
            "prompt": prompt
        }
