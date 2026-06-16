If your goal is to build a **real RAG project that you can put on your resume**, don't start with LangGraph, agents, memory, MCP, etc.

Build a solid RAG first.

---

# Architecture You Should Build

```text
                PDF Files
                    │
                    ▼
            Document Loader
                    │
                    ▼
               Chunking
                    │
                    ▼
              Embeddings
                    │
                    ▼
            Vector Database
                    │
     ┌──────────────┴──────────────┐
     │                             │
     ▼                             ▼
User Question              Stored Chunks
     │                             │
     └────► Similarity Search ◄────┘
                    │
                    ▼
          Top Relevant Chunks
                    │
                    ▼
            Prompt Builder
                    │
                    ▼
                 LLM
                    │
                    ▼
                 Answer
```

---

# Recommended Tech Stack

For learning:

```text
Frontend:
    Streamlit

Framework:
    LangChain

LLM:
    Ollama (Llama 3)

Embedding Model:
    nomic-embed-text
    OR
    bge-small-en

Vector Database:
    ChromaDB

PDF Reader:
    PyPDFLoader

Chunking:
    RecursiveCharacterTextSplitter
```

---

# Project Folder Structure

```text
rag_project/

│
├── app.py
│
├── data/
│   ├── pdf1.pdf
│   ├── pdf2.pdf
│
├── database/
│
├── ingest.py
│
├── rag.py
│
├── requirements.txt
│
└── utils/
    ├── loader.py
    ├── chunker.py
    ├── embeddings.py
```

---

# Step 1: Install Everything

```bash
pip install langchain
pip install langchain-community
pip install chromadb
pip install pypdf
pip install streamlit
pip install sentence-transformers
pip install ollama
```

---

# Step 2: Run Ollama

Install Ollama.

Pull model:

```bash
ollama pull llama3
```

Pull embedding model:

```bash
ollama pull nomic-embed-text
```

Check:

```bash
ollama list
```

You should see:

```text
llama3
nomic-embed-text
```

---

# Step 3: Load PDFs

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/manual.pdf")

docs = loader.load()
```

Output:

```text
Page 1
Page 2
Page 3
...
```

---

# Step 4: Chunking

Never store whole PDFs.

Use recursive chunking.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

---

Example:

PDF:

```text
10000 words
```

Becomes:

```text
Chunk 1
Chunk 2
Chunk 3
...
Chunk 50
```

---

# Why Chunk Overlap?

Without overlap:

```text
Chunk1:
Machine Learning is

Chunk2:
used in healthcare
```

Meaning lost.

---

With overlap:

```text
Chunk1:
Machine Learning is used

Chunk2:
used in healthcare
```

Context preserved.

---

# Step 5: Generate Embeddings

Convert chunks to vectors.

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

---

Chunk:

```text
Dogs are animals
```

becomes:

```text
[0.11, 0.44, -0.29, ...]
```

---

# Step 6: Store in ChromaDB

```python
from langchain.vectorstores import Chroma

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="database"
)

db.persist()
```

Now vectors are stored permanently.

---

# Step 7: User Query

User:

```text
What is leave policy?
```

---

Convert query to embedding.

```python
query_vector = embeddings.embed_query(
    "What is leave policy?"
)
```

---

# Step 8: Similarity Search

```python
results = db.similarity_search(
    "What is leave policy?",
    k=3
)
```

Retrieve:

```text
Chunk A
Chunk B
Chunk C
```

---

# Step 9: Build Prompt

```python
context = "\n".join(
    [doc.page_content for doc in results]
)

prompt = f"""
Context:

{context}

Question:

What is leave policy?
"""
```

---

# Step 10: Send to LLM

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

response = llm.invoke(prompt)
```

---

# That's Your First RAG

Flow:

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB

Question
 ↓
Retrieve Chunks
 ↓
Prompt
 ↓
Llama3
 ↓
Answer
```

---

# Phase 2 Improvements

After basic RAG works:

### Metadata

Store:

```text
PDF Name
Page Number
Source
```

Example:

```python
doc.metadata
```

---

### Source Citation

Output:

```text
Answer:
...

Source:
manual.pdf page 14
```

Very important.

---

### Hybrid Search

Combine:

```text
Keyword Search
+
Vector Search
```

Better retrieval.

---

### Re-ranking

Retrieve:

```text
Top 20 chunks
```

Then use a reranker model:

```text
bge-reranker
```

Keep best 5.

---

# Production RAG Architecture

```text
PDFs
  │
  ▼
Loader
  │
  ▼
Chunking
  │
  ▼
Embedding Model
  │
  ▼
Vector Database
  │
  ▼
Retriever
  │
  ▼
Reranker
  │
  ▼
Prompt Template
  │
  ▼
LLM
  │
  ▼
Answer + Sources
```

---

# Resume-Level Features

Add these:

### Multiple PDFs

```text
Upload many PDFs
```

### Chat History

```text
Conversation memory
```

### Source Citations

```text
Page numbers
```

### Streaming Responses

```text
Token by token output
```

### Local Models

```text
Ollama + Llama3
```

### Persistent Database

```text
ChromaDB
```

---

# What Interviewers Expect You To Know

If you build this project, be able to explain:

1. Why chunking is needed.
2. Why embeddings are needed.
3. Why vector DB is needed.
4. Difference between RAG and fine-tuning.
5. How similarity search works.
6. Why overlap is used.
7. What happens when a user asks a question.
8. How Ollama fits into the pipeline.
9. Why LLM is still needed after retrieval.
10. How to reduce hallucinations.

If you can explain those 10 points while demoing the project, you'll understand RAG better than most people who simply copy LangChain tutorials.

a