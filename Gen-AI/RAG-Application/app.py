import os
import streamlit as st
import logging
from ingest import (
    init_db, ingest_pdf, delete_document, get_all_documents,
    get_total_stats, get_chroma_vector_count, rebuild_database, get_chroma_db
)
from rag import list_ollama_models, query_rag, retrieve_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
init_db()

# Page setup
st.set_page_config(
    page_title="Semantic RAG Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphic Theme)
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #f3f4f6;
    }
    
    /* Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        text-shadow: 0px 4px 15px rgba(0, 242, 254, 0.15);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Metric Cards Grid */
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 30px;
        width: 100%;
    }
    
    .dashboard-card {
        flex: 1;
        min-width: 200px;
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        border-color: #00f2fe;
        box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
    }
    
    .card-title {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    .card-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00f2fe;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }
    
    /* Chunks Source Cards */
    .source-card {
        background: rgba(31, 41, 55, 0.5);
        border-left: 4px solid #00f2fe;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    .source-meta {
        font-size: 0.8rem;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    /* Alert details */
    .stAlert {
        background-color: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-title'>Semantic RAG Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Production-grade RAG interface built with Streamlit, SQLite, ChromaDB, and local Ollama models</p>", unsafe_allow_html=True)

# Fetch stats for sidebar and main dashboard
total_docs, total_chunks = get_total_stats()
vector_count = get_chroma_vector_count()

# Sidebar: Config & Info
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
    st.markdown("### ⚙️ System Configuration")
    
    # Model configuration
    ollama_models = list_ollama_models()
    selected_model = st.selectbox(
        "🧠 LLM Selection", 
        ollama_models,
        index=ollama_models.index("qwen2.5-coder:latest") if "qwen2.5-coder:latest" in ollama_models else 0,
        help="Select which local Ollama model to generate responses with."
    )
    
    st.markdown("---")
    st.markdown("### ✂️ Chunking Config")
    use_semantic = st.checkbox("🔮 Use Semantic Chunking", value=True, help="Enable sentence-similarity based chunking (recommends nomic-embed-text).")
    
    if use_semantic:
        similarity_threshold = st.slider(
            "Cosine Similarity Threshold", 
            min_value=0.3, max_value=0.9, value=0.65, step=0.05,
            help="Higher threshold yields smaller, more cohesive chunks. Lower threshold yields larger chunks."
        )
    else:
        similarity_threshold = 0.65
        
    chunk_size = st.number_input("Fallback Chunk Size", min_value=100, max_value=5000, value=1000, step=100)
    chunk_overlap = st.number_input("Fallback Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)

    st.markdown("---")
    st.markdown("### ℹ️ Engine Status")
    st.success("Embedding model: `nomic-embed-text`")
    st.info("SQLite registry: Active")

# Main Content: Tabs
tab1, tab2, tab3 = st.tabs(["💬 RAG Chat", "⚙️ Admin Dashboard", "🔍 Debug Console"])

# ----------------------------------------------------
# TAB 1: RAG Chat
# ----------------------------------------------------
with tab1:
    st.markdown("### Ask your documents")
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat messages
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                # Track toggle states
                meta_key = f"show_src_meta_{msg_idx}"
                text_key = f"show_src_text_{msg_idx}"
                if meta_key not in st.session_state:
                    st.session_state[meta_key] = False
                if text_key not in st.session_state:
                    st.session_state[text_key] = False
                
                # Place small buttons side-by-side on the bottom-right of the message
                col_space, col_btn1, col_btn2 = st.columns([3.8, 1.6, 1.6])
                
                with col_btn1:
                    btn_label1 = "🙈 Hide Sources" if st.session_state[meta_key] else "📄 View Sources"
                    if st.button(btn_label1, key=f"btn_src_{msg_idx}", use_container_width=True):
                        st.session_state[meta_key] = not st.session_state[meta_key]
                        st.rerun()
                        
                with col_btn2:
                    btn_label2 = "🙈 Hide Context" if st.session_state[text_key] else "🔍 View Context"
                    if st.button(btn_label2, key=f"btn_ctx_{msg_idx}", use_container_width=True):
                        st.session_state[text_key] = not st.session_state[text_key]
                        st.rerun()
                
                # Render metadata if toggled on
                if st.session_state[meta_key]:
                    st.markdown("##### 📄 Source Documents & Pages:")
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"- **Source [{idx+1}]**: `{src['filename']}` (Page {src['page']}) — *Similarity Score: {src['similarity_score']:.1f}%*")
                
                # Render text context if toggled on
                if st.session_state[text_key]:
                    st.markdown("##### 🔍 Retrieved Context Content:")
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-meta">Source [{idx+1}] - File: {src['filename']} | Page: {src['page']}</div>
                            <div>{src['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Chat input
    if prompt := st.chat_input("Enter your question here..."):
        # Display user question
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Ingestion guard
        if total_docs == 0:
            st.warning("Registry is empty. Please upload and index documents in the **Admin Dashboard** first.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🔍 Searching vector store...")
                
                # Fetch query answer
                with st.spinner("Analyzing context & generating response..."):
                    result = query_rag(prompt, llm_model=selected_model, k=4)
                    
                # Display answer
                message_placeholder.markdown(result["answer"])
                
                # Save chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
                st.rerun()

# ----------------------------------------------------
# TAB 2: Admin Dashboard
# ----------------------------------------------------
with tab2:
    # Key Stats Panel using beautiful CSS cards
    st.markdown(f"""
    <div class="card-container">
        <div class="dashboard-card">
            <div class="card-title">Total Documents</div>
            <div class="card-value">{total_docs}</div>
        </div>
        <div class="dashboard-card">
            <div class="card-title">Total Chunks</div>
            <div class="card-value">{total_chunks}</div>
        </div>
        <div class="dashboard-card">
            <div class="card-title">Vector Count</div>
            <div class="card-value">{vector_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # Left column: Upload form
    with col_left:
        st.markdown("### 📤 Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file to index", 
            type=["pdf"],
            help="Uploading a PDF with the same filename will replace the existing indexed document."
        )
        
        if uploaded_file is not None:
            # Save PDF file to data/
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            if st.button("🚀 Index Document", use_container_width=True):
                with st.status("Ingesting and indexing...", expanded=True) as status:
                    try:
                        status.update(label="Reading and parsing PDF...")
                        # Run ingestion
                        doc_id, count = ingest_pdf(
                            file_path, 
                            use_semantic=use_semantic, 
                            similarity_threshold=similarity_threshold,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        status.update(label=f"Ingestion successful! Created {count} chunks.", state="complete", expanded=False)
                        st.success(f"Indexed **{uploaded_file.name}** successfully.")
                        st.rerun()
                    except Exception as e:
                        status.update(label="Ingestion failed!", state="error", expanded=True)
                        st.error(f"Error: {e}")
                        
    # Right column: Document Management
    with col_right:
        st.markdown("### 📋 Document Registry")
        
        # Search document
        search_query = st.text_input("🔍 Search documents by filename", "")
        documents = get_all_documents(search_query)
        
        if not documents:
            st.info("No documents found in registry.")
        else:
            for doc in documents:
                # Create a card border for each document
                with st.container(border=True):
                    col_info, col_actions = st.columns([3, 2])
                    with col_info:
                        st.markdown(f"**📄 {doc['filename']}**")
                        st.markdown(f"<span style='font-size: 0.8rem; color:#9ca3af;'>Uploaded: {doc['upload_date']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='font-size: 0.8rem; color:#9ca3af;'>Chunks: {doc['chunk_count']} | Status: <b style='color:#00f2fe;'>{doc['indexing_status']}</b></span>", unsafe_allow_html=True)
                    
                    with col_actions:
                        # Spacing
                        st.write("")
                        col_reindex, col_del = st.columns(2)
                        
                        # Reindex button
                        if col_reindex.button("🔄 Re-index", key=f"reindex_{doc['document_id']}", use_container_width=True):
                            file_path = os.path.join("data", doc["filename"])
                            if os.path.exists(file_path):
                                with st.spinner("Re-indexing..."):
                                    try:
                                        ingest_pdf(
                                            file_path,
                                            use_semantic=use_semantic,
                                            similarity_threshold=similarity_threshold,
                                            chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap
                                        )
                                        st.success(f"Re-indexed {doc['filename']}.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error re-indexing: {e}")
                            else:
                                st.error("Physical PDF file missing from data folder!")
                                
                        # Delete button
                        if col_del.button("🗑️ Delete", key=f"del_{doc['document_id']}", use_container_width=True):
                            with st.spinner("Deleting..."):
                                delete_document(doc["document_id"])
                            st.success(f"Deleted {doc['filename']}.")
                            st.rerun()
                            
    st.markdown("---")
    st.markdown("### ⚠️ Danger Zone")
    st.warning("Completely purge the SQLite database, ChromaDB vector collection, and all physical documents.")
    
    # Reset dialog
    confirm_reset = st.checkbox("I understand this will delete all indexed documents and clear the database.")
    if st.button("🗑️ Rebuild / Reset Database", type="primary", disabled=not confirm_reset):
        with st.spinner("Purging all databases..."):
            rebuild_database()
        st.success("Database has been reset to a fresh state.")
        st.rerun()

# ----------------------------------------------------
# TAB 3: Debug Console
# ----------------------------------------------------
with tab3:
    st.markdown("### 🛠️ Debugging & Inspection Console")
    
    debug_opt = st.selectbox(
        "Select Debug View",
        ["Chunk & Metadata Viewer", "Retrieval Test Bench"]
    )
    
    if debug_opt == "Chunk & Metadata Viewer":
        st.markdown("#### Inspect chunks stored in ChromaDB")
        docs_list = get_all_documents()
        
        if not docs_list:
            st.info("No documents indexed. Nothing to inspect.")
        else:
            # Document selector
            selected_doc = st.selectbox(
                "Select Document to Inspect",
                docs_list,
                format_func=lambda x: f"{x['filename']} (Chunks: {x['chunk_count']})"
            )
            
            # Fetch document chunks
            db = get_chroma_db()
            results = db.get(where={"document_id": selected_doc["document_id"]})
            
            if results and "documents" in results and len(results["documents"]) > 0:
                chunks_data = []
                for idx, (doc_text, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                    chunks_data.append({
                        "text": doc_text,
                        "page": meta.get("page", 1),
                        "chunk_index": meta.get("chunk_index", 0),
                        "metadata": meta
                    })
                # Sort by index
                chunks_data.sort(key=lambda x: x["chunk_index"])
                
                # Display list
                st.markdown(f"Showing **{len(chunks_data)}** chunks:")
                for c in chunks_data:
                    with st.expander(f"📦 Chunk {c['chunk_index']} (Page {c['page']})", expanded=False):
                        st.text_area("Chunk Content", c["text"], height=120, disabled=True)
                        st.json(c["metadata"])
            else:
                st.warning("No chunks found in ChromaDB for this document. Try re-indexing it.")
                
    elif debug_opt == "Retrieval Test Bench":
        st.markdown("#### Test similarity search and check similarity scores")
        
        test_query = st.text_input("Enter Query to Test", "")
        test_k = st.slider("Number of Chunks to Retrieve (k)", min_value=1, max_value=10, value=4)
        
        if st.button("🔍 Run Retrieval Test") and test_query:
            with st.spinner("Searching vector store..."):
                retrieved = retrieve_context(test_query, k=test_k)
                
            if not retrieved:
                st.info("No matches found or database empty.")
            else:
                st.markdown(f"Retrieved **{len(retrieved)}** chunks:")
                for idx, item in enumerate(retrieved):
                    st.markdown(f"##### Match [{idx+1}] - Similarity: **{item['similarity_score']:.2f}%**")
                    col_info1, col_info2 = st.columns(2)
                    col_info1.write(f"**Filename:** {item['filename']}")
                    col_info2.write(f"**Page Number:** {item['page']} | **Chunk Index:** {item['chunk_index']}")
                    
                    st.text_area("Retrieved Chunk", item["content"], height=100, key=f"ret_text_{idx}", disabled=True)
                    st.markdown("---")
