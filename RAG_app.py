####################################################################
#  RAG-Powered AI Study Assistant
#  GenAI Stack:  LlamaIndex  +  FAISS  +  Ollama (Mistral / Qwen)  +  Streamlit
#
#  Data flow (the "GenAI Stack"):
#
#    1. INGEST    Upload docs ──► SimpleDirectoryReader (load)
#    2. CHUNK     ──► SentenceSplitter (chunk + overlap)
#    3. EMBED     ──► HuggingFace embeddings (local, no API)
#    4. STORE     ──► FAISS vector index (persisted to disk)
#    5. RETRIEVE  query ──► embed ──► FAISS top-k similarity search
#    6. GENERATE  context + question ──► Ollama LLM (Mistral / Qwen)
#                 ──► source-cited answer
#    7. SERVE     ──► Streamlit chat UI (with memory)
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import tempfile
from pathlib import Path

import faiss
import streamlit as st

# ---- LlamaIndex: ingestion, indexing, retrieval, generation ----
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore


####################################################################
#                           Config
####################################################################

# Open-source LLMs served locally by Ollama.
# Pull them first, e.g.:  `ollama pull mistral`  /  `ollama pull qwen2.5`
LIST_LLM_MODELS = [
    "mistral",
    "qwen2.5",
    "qwen2.5:3b",
    "llama3.1",
    "phi3",
]

# Local HuggingFace embedding models (downloaded once, run on CPU).
LIST_EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",   # 384-dim, fast, great default
    "thenlper/gte-large",       # 1024-dim, higher quality, heavier
    "BAAI/bge-base-en-v1.5",    # 768-dim, balanced
]

WELCOME_MESSAGE = "👋 Hi! Upload your course materials, build a vectorstore, then ask me anything about them."

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR.joinpath("data", "vector_stores")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Context prompt that instructs the LLM to answer *only* from retrieved
# context and to cite its sources — this is what minimises hallucinations.
CONTEXT_PROMPT = (
    "You are a study assistant answering questions about technical course "
    "materials. Use ONLY the context below to answer. If the answer is not "
    "in the context, say you don't know — do not make things up.\n"
    "Always ground your answer in the provided sources.\n"
    "----------------------\n"
    "{context_str}\n"
    "----------------------\n"
)


####################################################################
#                  Cached / shared resources
####################################################################
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    """Load a local HuggingFace embedding model (cached across reruns)."""
    return HuggingFaceEmbedding(model_name=model_name)


def get_embedding_dim(embed_model: HuggingFaceEmbedding) -> int:
    """Probe the embedding dimension so we can size the FAISS index."""
    return len(embed_model.get_text_embedding("dimension probe"))


def build_llm() -> Ollama:
    """Instantiate the Ollama-served open-source LLM (Mistral / Qwen / ...)."""
    return Ollama(
        model=st.session_state.selected_model,
        request_timeout=st.session_state.request_timeout,
        temperature=st.session_state.temperature,
    )


####################################################################
#          1-4. Ingestion pipeline -> FAISS vector store
####################################################################
def build_index_from_files(file_paths, persist_dir, embed_model):
    """The ingestion pipeline: load -> chunk -> embed -> store in FAISS.

    Parameters:
        file_paths   : list of paths to the uploaded documents.
        persist_dir  : folder where the FAISS index + docstore are saved.
        embed_model  : local HuggingFace embedding model.
    """
    # 1. LOAD documents (pdf / txt / docx / csv / md ...).
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    # 2. CHUNK into overlapping nodes.
    splitter = SentenceSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # 3 & 4. EMBED each node and STORE the vectors in a FAISS index.
    dim = get_embedding_dim(embed_model)
    faiss_index = faiss.IndexFlatL2(dim)  # L2 (euclidean) flat index
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    # Persist the FAISS index + docstore to disk for later reuse.
    index.storage_context.persist(persist_dir=persist_dir)
    return index, len(documents), len(nodes)


def load_index_from_disk(persist_dir, embed_model):
    """Reload a previously persisted FAISS index from disk."""
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_dir
    )
    return load_index_from_storage(storage_context, embed_model=embed_model)


####################################################################
#       5 & 6. Retrieval + generation -> conversational engine
####################################################################
def build_chat_engine(index, llm):
    """Create a conversational engine that retrieves from FAISS, keeps
    chat memory, and generates source-grounded answers with the LLM."""
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=memory,
        context_prompt=CONTEXT_PROMPT,
        similarity_top_k=st.session_state.similarity_top_k,
        verbose=False,
    )


def reset_engine_from_index():
    """(Re)build the chat engine using the current index + LLM settings."""
    st.session_state.chat_engine = build_chat_engine(
        st.session_state.index, build_llm()
    )
    clear_chat_history()


####################################################################
#                       Streamlit sidebar
####################################################################
def sidebar():
    with st.sidebar:
        st.caption(
            "🚀 RAG study assistant — **LlamaIndex · FAISS · Ollama "
            "(Mistral/Qwen) · Streamlit**"
        )
        st.divider()

        st.subheader("🧠 Language model (Ollama)")
        st.session_state.selected_model = st.selectbox(
            "Open-source LLM", LIST_LLM_MODELS
        )
        st.caption(
            "Served locally by [Ollama](https://ollama.com). "
            "Pull first, e.g. `ollama pull mistral`."
        )
        st.session_state.temperature = st.slider(
            "temperature", 0.0, 1.0, 0.1, 0.05
        )
        st.session_state.request_timeout = st.slider(
            "request timeout (s)", 30, 600, 180, 30
        )

        st.divider()
        st.subheader("🔎 Embeddings & retrieval")
        st.session_state.embedding_model_name = st.selectbox(
            "Embedding model (local HF)", LIST_EMBEDDING_MODELS
        )
        st.session_state.chunk_size = st.slider(
            "chunk size", 256, 2048, 1024, 64
        )
        st.session_state.chunk_overlap = st.slider(
            "chunk overlap", 0, 512, 200, 20
        )
        st.session_state.similarity_top_k = st.slider(
            "retrieved chunks (top-k)", 1, 12, 4, 1
        )


####################################################################
#               Document chooser (create / load store)
####################################################################
def document_chooser():
    tab_create, tab_open = st.tabs(
        ["📥 Create a new Vectorstore", "📂 Open a saved Vectorstore"]
    )

    # ---------- Create ----------
    with tab_create:
        uploaded_files = st.file_uploader(
            "**Select course documents**",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv", "md"],
        )
        store_name = st.text_input(
            "**Vectorstore name** — docs will be loaded, chunked, embedded "
            "and ingested into a FAISS index.",
            placeholder="e.g. ml_course",
        )

        if st.button("Create Vectorstore", type="primary"):
            if not uploaded_files:
                st.warning("Please select at least one document.")
            elif not store_name.strip():
                st.warning("Please provide a vectorstore name.")
            else:
                create_vectorstore(uploaded_files, store_name.strip())

    # ---------- Open ----------
    with tab_open:
        existing = sorted(
            p.name for p in VECTOR_STORE_DIR.iterdir() if p.is_dir()
        )
        if not existing:
            st.info("No saved vectorstores yet. Create one first.")
        else:
            chosen = st.selectbox("Select a vectorstore", existing)
            if st.button("Load Vectorstore"):
                load_vectorstore(chosen)


def create_vectorstore(uploaded_files, store_name):
    """Persist uploaded files to a temp dir, run the ingestion pipeline,
    and build the conversational engine."""
    persist_dir = VECTOR_STORE_DIR.joinpath(store_name).as_posix()

    with st.spinner("Loading, chunking, embedding and indexing documents..."):
        try:
            embed_model = load_embedding_model(
                st.session_state.embedding_model_name
            )

            # Write uploads to a temporary directory for the reader.
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    fp = os.path.join(tmp_dir, uploaded_file.name)
                    with open(fp, "wb") as f:
                        f.write(uploaded_file.read())
                    file_paths.append(fp)

                index, n_docs, n_chunks = build_index_from_files(
                    file_paths, persist_dir, embed_model
                )

            st.session_state.index = index
            reset_engine_from_index()
            st.success(
                f"Vectorstore **{store_name}** created — "
                f"{n_docs} document(s), {n_chunks} chunk(s) indexed in FAISS."
            )
        except Exception as e:
            st.error(f"Failed to create vectorstore: {e}")


def load_vectorstore(store_name):
    persist_dir = VECTOR_STORE_DIR.joinpath(store_name).as_posix()
    with st.spinner("Loading FAISS vectorstore..."):
        try:
            embed_model = load_embedding_model(
                st.session_state.embedding_model_name
            )
            st.session_state.index = load_index_from_disk(
                persist_dir, embed_model
            )
            reset_engine_from_index()
            st.success(f"**{store_name}** loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load vectorstore: {e}")


####################################################################
#                         Chat helpers
####################################################################
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]
    engine = st.session_state.get("chat_engine")
    if engine is not None:
        engine.reset()


def render_source_documents(source_nodes):
    """Show the retrieved chunks (source citations) below the answer."""
    if not source_nodes:
        return
    with st.expander("📚 **Source documents**"):
        content = ""
        for i, node in enumerate(source_nodes, start=1):
            meta = node.node.metadata or {}
            source = meta.get("file_name") or meta.get("file_path") or "unknown"
            page = meta.get("page_label")
            page_str = f" (page {page})" if page else ""
            score = f" · score={node.score:.3f}" if node.score is not None else ""
            content += f"**[{i}] {source}{page_str}{score}**\n\n"
            content += node.node.get_content().strip() + "\n\n---\n\n"
        st.markdown(content)


def answer_question(prompt):
    """Run retrieval + generation and render the cited answer."""
    try:
        response = st.session_state.chat_engine.chat(prompt)
        answer = str(response.response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st.markdown(answer)
            render_source_documents(response.source_nodes)
    except Exception as e:
        st.warning(f"Error while answering: {e}")


####################################################################
#                            App
####################################################################
def main():
    st.set_page_config(page_title="RAG Study Assistant", page_icon="🤖")
    st.title("🤖 RAG-Powered AI Study Assistant")

    sidebar()
    document_chooser()
    st.divider()

    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("💬 Chat with your course materials")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if "chat_engine" not in st.session_state:
            st.info("Please create or load a vectorstore first.")
            st.stop()
        with st.spinner("Retrieving context and generating answer..."):
            answer_question(prompt)


if __name__ == "__main__":
    main()
