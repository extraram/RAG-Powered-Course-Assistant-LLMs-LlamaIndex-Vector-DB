# RAG-Powered AI Study Assistant

**LLMs · Vector Databases · LlamaIndex**

A Retrieval-Augmented Generation (RAG) chatbot that gives context-aware,
**source-cited** answers from technical course materials — built on a modern,
fully open-source **GenAI stack** that runs locally with no paid APIs.

---

## Project Overview

Large Language Models are powerful but hallucinate and go stale, because they
only know their training data. This assistant grounds an open-source LLM in your
own documents using **Retrieval-Augmented Generation**: your files are chunked,
embedded and indexed, the most relevant passages are retrieved for each question,
and the LLM answers using *only* that retrieved context — citing its sources.

- **Scalable ingestion pipeline** built with **LlamaIndex** to load, chunk and
  embed documents.
- **Efficient vector similarity search** powered by **FAISS** (local, file-based,
  swappable for Weaviate).
- **Open-source LLMs (Mistral / Qwen)** served locally via **Ollama** for
  accurate, source-cited responses that minimise hallucinations.
- **Responsive UI** built with **Streamlit** so students can interact with
  complex course datasets in real time.

## The GenAI Stack — data flow

```
        ┌──────────────┐   load    ┌──────────────┐  chunk   ┌──────────────┐
Upload  │ SimpleDirectory│ ───────► │ SentenceSplit│ ───────► │  HuggingFace │
docs ──►│    Reader      │          │   (chunk +   │          │  embeddings  │
        └──────────────┘          │   overlap)   │          │  (local CPU) │
                                    └──────────────┘          └──────┬───────┘
                                                                     │ embed
                                                                     ▼
   ┌───────────────────────────────────────────────┐         ┌──────────────┐
   │  Ollama LLM (Mistral / Qwen)                   │ context │     FAISS    │
   │  "answer using ONLY this context, cite sources"│ ◄───────│ vector index │
   └───────────────────────┬───────────────────────┘ top-k   │  (on disk)   │
                           │ source-cited answer      retrieve└──────▲───────┘
                           ▼                                         │ embed query
                 ┌──────────────────┐                                │
                 │  Streamlit chat  │ ◄──────────────  user question ┘
                 │   (with memory)  │
                 └──────────────────┘
```

| Stage | Component | Library |
|-------|-----------|---------|
| 1. Load | `SimpleDirectoryReader` | LlamaIndex |
| 2. Chunk | `SentenceSplitter` (size + overlap) | LlamaIndex |
| 3. Embed | `HuggingFaceEmbedding` (e.g. `bge-small-en-v1.5`) | LlamaIndex + sentence-transformers |
| 4. Store | `FaissVectorStore` (`IndexFlatL2`) persisted to disk | FAISS |
| 5. Retrieve | top-k similarity search | FAISS |
| 6. Generate | `Ollama` (Mistral / Qwen) + condense-plus-context engine | LlamaIndex + Ollama |
| 7. Serve | chat UI with conversational memory & source citations | Streamlit |

## Installation

Requires **Python 3.9+** and a running **[Ollama](https://ollama.com)** instance.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama and pull an open-source model
#    (download from https://ollama.com, then:)
ollama pull mistral
ollama pull qwen2.5
```

## Run the app

```bash
streamlit run RAG_app.py
```

Then in the browser:

1. In the sidebar, pick an **LLM** (Mistral / Qwen / …), an **embedding model**,
   and retrieval parameters (chunk size, overlap, top-k).
2. On the **Create a new Vectorstore** tab, upload documents
   (`pdf`, `txt`, `docx`, `csv`, `md`), name the store, and click
   **Create Vectorstore** — this runs the ingestion pipeline and builds a FAISS
   index that is persisted to `data/vector_stores/`.
3. Or reload a previously built index from the **Open a saved Vectorstore** tab.
4. **Chat with your data** — every answer comes with an expandable
   **Source documents** panel showing the retrieved chunks and similarity scores.

## Why this minimises hallucinations

The generation prompt instructs the model to answer **only** from retrieved
context and to say "I don't know" when the answer isn't present. Because the
answer is grounded in the top-k FAISS matches — and those exact chunks are shown
to the user as citations — responses stay faithful to the source material instead
of being invented by the LLM.

## Swapping components

The stack is modular:

- **Vector store** — replace `FaissVectorStore` with a Weaviate vector store for
  a client/server deployment.
- **LLM** — point Ollama at any local model, or swap in a hosted inference API.
- **Embeddings** — choose any `sentence-transformers` model in the sidebar.
