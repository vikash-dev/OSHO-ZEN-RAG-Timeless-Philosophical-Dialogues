import os
import sys
import time
import re
import uuid
import logging
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

import torch
import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OshoRAG")

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
INDEX_PATH_PRIMARY = "./OSHO_Philosophy_Master_Index"
INDEX_PATH_FALLBACK = "./OSHO_Master_Index"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVAL_K = 3
MAX_HISTORY_TURNS = 6  # 3 user + 3 assistant turns

# Global runtime state
class AppState:
    vectorstore: Optional[FAISS] = None
    retriever: Any = None
    embedding_model: Optional[HuggingFaceEmbeddings] = None
    total_vectors: int = 0
    resolved_index_path: str = ""
    is_ready: bool = False
    load_time_seconds: float = 0.0
    device: str = DEVICE

state = AppState()

# In-memory session history store: session_id -> list of {"role": "user"|"assistant", "content": str}
chat_sessions: Dict[str, List[Dict[str, str]]] = {}


# ==========================================
# VECTOR STORE INITIALIZATION
# ==========================================
def initialize_vectorstore():
    """Loads embedding model and FAISS vector index into memory."""
    t0 = time.time()
    logger.info(f"🚀 Initializing HuggingFace Embeddings ('{EMBEDDING_MODEL_NAME}') on {DEVICE}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    state.embedding_model = embeddings

    # Resolve index path
    target_index = INDEX_PATH_PRIMARY
    if not os.path.exists(target_index):
        if os.path.exists(INDEX_PATH_FALLBACK):
            logger.warning(f"Primary index path '{INDEX_PATH_PRIMARY}' not found. Falling back to '{INDEX_PATH_FALLBACK}'.")
            target_index = INDEX_PATH_FALLBACK
        else:
            raise FileNotFoundError(f"Neither '{INDEX_PATH_PRIMARY}' nor '{INDEX_PATH_FALLBACK}' could be located.")

    logger.info(f"📂 Loading FAISS Master Index from '{target_index}' (allow_dangerous_deserialization=True)...")
    vectorstore = FAISS.load_local(
        target_index,
        embeddings,
        allow_dangerous_deserialization=True
    )

    state.vectorstore = vectorstore
    state.retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    state.total_vectors = getattr(vectorstore.index, "ntotal", 0)
    state.resolved_index_path = target_index
    state.load_time_seconds = round(time.time() - t0, 2)
    state.is_ready = True

    logger.info(f"✅ Vector database successfully loaded! Total vectors: {state.total_vectors:,} in {state.load_time_seconds}s")


# ==========================================
# LIFESPAN & FASTAPI APP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load embeddings and index
    try:
        initialize_vectorstore()
    except Exception as e:
        logger.error(f"❌ Failed to load vector store on startup: {e}", exc_info=True)
    yield
    # Shutdown
    logger.info("🛑 Shutting down Osho Philosophy RAG Chatbot.")

app = FastAPI(
    title="Osho Philosophy RAG Chatbot",
    description="Zen-inspired RAG Web Application backed by FAISS and local Ollama (qwen2.5:3b)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class CitationItem(BaseModel):
    book: str
    filename: str
    page: Any
    parent_id: Optional[Any] = None
    snippet: str
    parent_content: str
    score: Optional[float] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Seeker's question or prompt")
    session_id: Optional[str] = Field(default="default", description="Conversation session ID")

class ChatResponse(BaseModel):
    answer: str
    sources: List[CitationItem]
    latency: float
    session_id: str
    model: str

class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"

class ResetResponse(BaseModel):
    status: str
    message: str
    session_id: str

class StatsResponse(BaseModel):
    status: str
    model: str
    vector_count: int
    embedding_model: str
    index_path: str
    device: str
    retrieval_k: int
    load_time_seconds: float
    sessions_active: int


# ==========================================
# RAG CORE UTILITIES
# ==========================================
def clean_book_title(filename: str) -> str:
    """Removes extension and cleans up filename for elegant display."""
    if not filename:
        return "Discourse Transcript"
    base = os.path.basename(filename)
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_\-]+", " ", base)
    return base.strip()

def build_osho_prompt(context_blocks: List[str], history: List[Dict[str, str]], query: str) -> str:
    """
    Constructs an enlightened, nuanced prompt adhering to Osho's philosophical voice,
    grounded firmly in the retrieved parent chunks.
    """
    system_instruction = (
        "You are an enlightened, compassionate, and poetic philosophical guide deeply immersed in the discourses of Osho.\n"
        "Your mission is to share Osho's timeless perspective on meditation, consciousness, freedom, love, the ego, silence, and the celebration of life.\n\n"
        "CORE GUIDELINES:\n"
        "1. GROUNDING: Base your philosophical response directly and authentically on the provided Discourse Excerpts below. "
        "Do not invent factual details or cite external ideas outside of Osho's vision.\n"
        "2. VOICE & TONE: Speak in Osho's unmistakable voice — direct, evocative, luminous, playful, and paradoxical. "
        "Inspire the seeker toward self-awareness, witnessing (Sakshi), and living in the eternal present moment.\n"
        "3. HONESTY: If the provided excerpts do not contain enough specific information to answer a factual inquiry, gently explain what is "
        "and isn't touched upon in the available passages, while offering whatever contemplative insight aligns with Osho's teachings.\n"
        "4. FORMATTING: Use clean Markdown with paragraphs and gentle emphasis. Avoid robotic phrases like 'According to the context'."
    )

    context_str = "\n\n---\n\n".join(context_blocks)

    # Format history turns (last few)
    history_str = ""
    if history:
        history_lines = []
        for msg in history[-MAX_HISTORY_TURNS:]:
            role_label = "Seeker" if msg.get("role") == "user" else "Osho"
            history_lines.append(f"{role_label}: {msg.get('content', '').strip()}")
        history_str = "### Dialogue History:\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""{system_instruction}

### Provided Discourse Excerpts from Osho's Books:
{context_str}

{history_str}### Seeker's Inquiry:
{query}

### Osho's Contemplative Response:"""
    return prompt


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the Zen dark-mode RAG Web UI."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Index template not found.")
    return FileResponse(template_path, media_type="text/html")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Returns real-time status and telemetry for the UI sidebar."""
    return StatsResponse(
        status="Ready" if state.is_ready else "Initializing",
        model=OLLAMA_MODEL,
        vector_count=state.total_vectors,
        embedding_model=EMBEDDING_MODEL_NAME,
        index_path=state.resolved_index_path or INDEX_PATH_PRIMARY,
        device=state.device,
        retrieval_k=RETRIEVAL_K,
        load_time_seconds=state.load_time_seconds,
        sessions_active=len(chat_sessions)
    )


@app.get("/api/health")
async def health_check():
    """Health check for service monitoring."""
    return {
        "status": "healthy" if state.is_ready else "starting",
        "index_ready": state.is_ready,
        "total_vectors": state.total_vectors
    }


@app.post("/api/reset", response_model=ResetResponse)
async def reset_chat(req: ResetRequest):
    """Clears conversation history for the given session ID."""
    session_id = req.session_id or "default"
    if session_id in chat_sessions:
        chat_sessions[session_id].clear()
    else:
        chat_sessions[session_id] = []
    
    logger.info(f"🧹 Session history cleared for: {session_id}")
    return ResetResponse(
        status="ok",
        message="Contemplation history has been refreshed and cleared.",
        session_id=session_id
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main RAG pipeline:
    1. Hierarchical parent-child retrieval from FAISS (k=3)
    2. Prompt formatting with context and session history
    3. Ollama inference with latency measurement
    4. Memory update and structured response return
    """
    if not state.is_ready or state.vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector database is currently initializing. Please try again in a few moments."
        )

    user_query = req.message.strip()
    session_id = req.session_id or "default"
    if not user_query:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    start_time = time.time()

    # 1. Hierarchical Parent-Child Retrieval (k=3)
    try:
        # We use similarity_search_with_score to retrieve the closest child vectors
        docs_and_scores = state.vectorstore.similarity_search_with_score(user_query, k=RETRIEVAL_K)
    except Exception as e:
        logger.error(f"Error during FAISS retrieval: {e}", exc_info=True)
        # Fallback to standard retriever
        raw_docs = state.retriever.invoke(user_query)
        docs_and_scores = [(d, 0.0) for d in raw_docs]

    context_blocks: List[str] = []
    citations: List[CitationItem] = []

    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata or {}
        raw_source = meta.get("source", "Osho Discourse.pdf")
        clean_title = clean_book_title(raw_source)
        page_num = meta.get("page", 1)
        parent_id = meta.get("parent_id")
        
        # Hierarchical retrieval: preferentially use rich parent_content
        parent_content = meta.get("parent_content")
        if parent_content and isinstance(parent_content, str) and parent_content.strip():
            rich_text = parent_content.strip()
        else:
            rich_text = doc.page_content.strip()

        # Build prompt context block (optimized length for responsive local CPU inference)
        prompt_snippet = rich_text[:1400] + ("..." if len(rich_text) > 1400 else "")
        block = (
            f"[Discourse {idx} | Book: '{clean_title}', Page: {page_num}]\n"
            f"{prompt_snippet}"
        )
        context_blocks.append(block)

        # Build citation for UI accordion (preserves full parent content)
        snippet_preview = doc.page_content.strip()
        if len(snippet_preview) > 320:
            snippet_preview = snippet_preview[:320] + "..."

        citations.append(
            CitationItem(
                book=clean_title,
                filename=os.path.basename(raw_source),
                page=page_num,
                parent_id=parent_id,
                snippet=snippet_preview,
                parent_content=rich_text,
                score=round(float(score), 4) if score is not None else None
            )
        )

    # 2. Retrieve session history
    session_history = chat_sessions.setdefault(session_id, [])

    # 3. Construct prompt
    prompt = build_osho_prompt(context_blocks, session_history, user_query)

    # 4. Query Ollama local API
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 350
        }
    }

    answer_text = ""
    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(OLLAMA_API_URL, json=payload)
            if resp.status_code == 200:
                answer_text = resp.json().get("response", "").strip()
            else:
                logger.error(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                answer_text = (
                    "Osho's contemplation could not be fully transmitted because the local language model "
                    f"returned an error ({resp.status_code}). Please verify that Ollama is active with `{OLLAMA_MODEL}`."
                )
    except httpx.ConnectError:
        logger.error(f"Could not connect to Ollama at {OLLAMA_API_URL}")
        answer_text = (
            "⚠️ **Could not connect to local Ollama server**.\n\n"
            f"Please make sure Ollama is running (`ollama serve`) and that `{OLLAMA_MODEL}` is loaded."
        )
    except (httpx.TimeoutException, httpx.ReadTimeout):
        logger.error("Inference timed out waiting for Ollama response.")
        answer_text = (
            "⏳ **Contemplation Timed Out**: The local CPU is taking longer than usual to complete the inference. "
            "Try re-asking or shortening the inquiry, or check CPU utilization."
        )
    except Exception as e:
        logger.error(f"Inference exception: {e}", exc_info=True)
        answer_text = f"An unexpected error occurred while communicating with the reasoning model: {str(e)}"

    elapsed_time = round(time.time() - start_time, 2)

    # 5. Update session memory (user query + model answer)
    session_history.append({"role": "user", "content": user_query})
    session_history.append({"role": "assistant", "content": answer_text})

    # Keep memory bounded
    if len(session_history) > MAX_HISTORY_TURNS * 2:
        chat_sessions[session_id] = session_history[-MAX_HISTORY_TURNS * 2:]

    logger.info(f"✨ Query completed in {elapsed_time}s | Session: {session_id} | Chunks: {len(citations)}")

    return ChatResponse(
        answer=answer_text,
        sources=citations,
        latency=elapsed_time,
        session_id=session_id,
        model=OLLAMA_MODEL
    )


# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
