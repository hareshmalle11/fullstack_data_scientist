import os
import shutil
import sqlite3
import uuid
import logging
from datetime import datetime
from langchain_community.vectorstores import Chroma
from utils.loader import load_pdf
from utils.chunker import chunk_document
from utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "registry.db")
DATA_DIR = "data"
CHROMA_DIR = os.path.join(DB_DIR, "chroma")

# Ensure required directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------
# SQLite Database Layer
# ----------------------------------------------------

def init_db():
    """Initializes the SQLite database and creates the document registry table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT UNIQUE,
            upload_date TEXT,
            chunk_count INTEGER,
            indexing_status TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized successfully.")

def add_document_record(doc_id: str, filename: str, chunk_count: int, status: str):
    """Inserts a new document record into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO documents (document_id, filename, upload_date, chunk_count, indexing_status)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, filename, upload_date, chunk_count, status))
        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting SQLite record for {filename}: {e}")
        raise e
    finally:
        conn.close()

def update_document_status(doc_id: str, status: str, chunk_count: int = None):
    """Updates the status and optional chunk count of a document."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        if chunk_count is not None:
            cursor.execute("""
                UPDATE documents SET indexing_status = ?, chunk_count = ? WHERE document_id = ?
            """, (status, chunk_count, doc_id))
        else:
            cursor.execute("""
                UPDATE documents SET indexing_status = ? WHERE document_id = ?
            """, (status, doc_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating SQLite status for document ID {doc_id}: {e}")
        raise e
    finally:
        conn.close()

def delete_document_record(doc_id: str):
    """Removes a document record from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
        conn.commit()
    except Exception as e:
        logger.error(f"Error deleting SQLite record for document ID {doc_id}: {e}")
        raise e
    finally:
        conn.close()

def get_all_documents(search_query: str = None):
    """Retrieves all documents from the registry, filtered by search query if provided."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        if search_query:
            cursor.execute("""
                SELECT document_id, filename, upload_date, chunk_count, indexing_status 
                FROM documents 
                WHERE filename LIKE ? 
                ORDER BY upload_date DESC
            """, (f"%{search_query}%",))
        else:
            cursor.execute("""
                SELECT document_id, filename, upload_date, chunk_count, indexing_status 
                FROM documents 
                ORDER BY upload_date DESC
            """)
        rows = cursor.fetchall()
        docs = []
        for r in rows:
            docs.append({
                "document_id": r[0],
                "filename": r[1],
                "upload_date": r[2],
                "chunk_count": r[3],
                "indexing_status": r[4]
            })
        return docs
    finally:
        conn.close()

def get_document_by_filename(filename: str):
    """Checks if a document with a given filename exists in the registry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT document_id, filename, upload_date, chunk_count, indexing_status 
            FROM documents WHERE filename = ?
        """, (filename,))
        row = cursor.fetchone()
        if row:
            return {
                "document_id": row[0],
                "filename": row[1],
                "upload_date": row[2],
                "chunk_count": row[3],
                "indexing_status": row[4]
            }
        return None
    finally:
        conn.close()

def get_total_stats():
    """Returns database summary stats: total documents and total chunk count."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*), SUM(chunk_count) FROM documents")
        row = cursor.fetchone()
        total_docs = row[0] if row[0] else 0
        total_chunks = row[1] if row[1] else 0
        return total_docs, total_chunks
    finally:
        conn.close()


# ----------------------------------------------------
# Vector DB & Ingestion Orchestration
# ----------------------------------------------------

def get_chroma_db():
    """Returns the LangChain Chroma wrapper initialized with our settings."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_documents"
    )

def ingest_pdf(file_path: str, use_semantic: bool = True, similarity_threshold: float = 0.65, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Orchestrates the loading, chunking, embedding, and storage of a PDF.
    Supports updating a document if the filename already exists.
    """
    init_db()
    filename = os.path.basename(file_path)
    
    # Check if file already exists in database
    existing_doc = get_document_by_filename(filename)
    if existing_doc:
        logger.info(f"Document {filename} already exists. Deleting older version before re-ingesting.")
        delete_document(existing_doc["document_id"])
        
    # Generate new document details
    doc_id = str(uuid.uuid4())
    add_document_record(doc_id, filename, 0, "Indexing")
    
    try:
        # 1. Load PDF
        pages = load_pdf(file_path)
        
        # 2. Chunk pages
        embeddings = get_embeddings()
        chunks = chunk_document(
            pages, 
            doc_id, 
            filename, 
            embeddings_model=embeddings, 
            use_semantic=use_semantic, 
            similarity_threshold=similarity_threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not chunks:
            raise ValueError("No text extracted from PDF. Document might be empty or scanned images.")
            
        # 3. Add to ChromaDB
        db = get_chroma_db()
        db.add_documents(chunks)
        
        # 4. Update SQLite status to Indexed
        update_document_status(doc_id, "Indexed", len(chunks))
        logger.info(f"Ingested document {filename} successfully with {len(chunks)} chunks.")
        return doc_id, len(chunks)
        
    except Exception as e:
        logger.error(f"Failed to ingest document {filename}: {e}")
        update_document_status(doc_id, "Failed")
        raise e

def delete_document(doc_id: str):
    """
    Deletes all vectors of a document from ChromaDB, removes the SQLite record,
    and deletes the local PDF file from the data folder.
    """
    # 1. Fetch document details
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM documents WHERE document_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        logger.warning(f"Document ID {doc_id} not found in registry.")
        return
        
    filename = row[0]
    
    # 2. Delete from ChromaDB
    try:
        db = get_chroma_db()
        # ChromaDB allows filtering deletes using a dictionary metadata where clause
        db._collection.delete(where={"document_id": doc_id})
        logger.info(f"Deleted vectors for {filename} (ID: {doc_id}) from ChromaDB.")
    except Exception as e:
        logger.error(f"Error deleting Chroma vectors for document ID {doc_id}: {e}")
        
    # 3. Delete from SQLite
    delete_document_record(doc_id)
    
    # 4. Delete file from data/ directory
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted physical file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete physical file {file_path}: {e}")

def get_chroma_vector_count():
    """Returns the total number of vectors in ChromaDB."""
    try:
        db = get_chroma_db()
        return db._collection.count()
    except Exception as e:
        logger.error(f"Error reading vector count from ChromaDB: {e}")
        return 0

def rebuild_database():
    """
    Completely purges the vector store, resets the SQLite registry,
    and removes all uploaded documents from the data directory.
    """
    init_db()
    logger.info("Initiating database purge...")
    
    # 1. Clear SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    
    # 2. Clear ChromaDB directory
    if os.path.exists(CHROMA_DIR):
        try:
            # We close any active client handles by letting Chroma go out of scope, 
            # then remove the directory completely.
            shutil.rmtree(CHROMA_DIR)
            logger.info("ChromaDB vector directory purged successfully.")
        except Exception as e:
            logger.error(f"Failed to delete Chroma directory {CHROMA_DIR}: {e}")
            
    # 3. Clear data directory files (except gitkeep)
    for f in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, f)
        if os.path.isfile(file_path) and f not in [".gitkeep"]:
            try:
                os.remove(file_path)
                logger.info(f"Deleted data file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")
                
    init_db()
    logger.info("Database rebuilding completed. Fresh database ready.")