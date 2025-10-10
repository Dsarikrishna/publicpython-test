# grocery_chatbot.py
# Streamlit grocery chatbot using LangGraph + Qdrant + Redis cache + OpenAI
# Ingests: (1) Local PDF, (2) Google Drive file by ID
#
# ─────────────────────────────────────────────────────────────────────────────
# ENV (.env):
#   OPENAI_API_KEY=...
#   QDRANT_URL=... (or use QDRANT_PATH for local embedded)
#   QDRANT_API_KEY=...
#   QDRANT_COLLECTION=mukti_kitchenware
#   REDIS_URL=redis://localhost:6379/0
#
# First run will: download Drive file → extract text → chunk → embed → upsert to Qdrant.
# Subsequent runs skip ingestion if collection already populated.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import time
import uuid
import math
from dataclasses import dataclass
from typing import List, Dict, Any, TypedDict, Optional

import streamlit as st
from dotenv import load_dotenv

# Vector & cache
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Optional Redis cache
try:
    import redis  # type: ignore
    HAS_REDIS = True
except Exception:
    redis = None  # type: ignore
    HAS_REDIS = False

# LLM + embeddings
from openai import OpenAI
import tiktoken

# PDF + Drive
import fitz  # PyMuPDF
import gdown

# LangGraph
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# Optional inline secrets: create a secrets_local.py with variables below to bypass envs
try:
    from secrets_local import (
        OPENAI_API_KEY as OPENAI_KEY_INLINE,
        QDRANT_URL as QDRANT_URL_INLINE,
        QDRANT_API_KEY as QDRANT_KEY_INLINE,
        QDRANT_PATH as QDRANT_PATH_INLINE,
        REDIS_URL as REDIS_URL_INLINE,
    )
except Exception:
    OPENAI_KEY_INLINE = None
    QDRANT_URL_INLINE = None
    QDRANT_KEY_INLINE = None
    QDRANT_PATH_INLINE = None
    REDIS_URL_INLINE = None

OPENAI_API_KEY = (
    (OPENAI_KEY_INLINE or os.getenv("OPENAI_API_KEY", ""))
    or ""
).strip().strip('"').strip("'")
if not OPENAI_API_KEY:
    st.stop()  # Streamlit-friendly halt with a visible error
client = OpenAI(api_key=OPENAI_API_KEY)

QDRANT_URL = (QDRANT_URL_INLINE or os.getenv("QDRANT_URL"))
QDRANT_API_KEY = (QDRANT_KEY_INLINE or os.getenv("QDRANT_API_KEY"))
QDRANT_PATH = (QDRANT_PATH_INLINE or os.getenv("QDRANT_PATH"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mukti_kitchenware")

REDIS_URL = (REDIS_URL_INLINE or os.getenv("REDIS_URL", "redis://localhost:6379/0"))

# Your files
LOCAL_PDF = r"C:\Users\bharg\OneDrive\Desktop\sap\mukti-kitchenware.pdf"
DRIVE_URL = "https://drive.google.com/file/d/1R8_C-WYAXSoCIxsXUkc7z77oBLaGUjEV/view?usp=drive_link"
DRIVE_ID = "1R8_C-WYAXSoCIxsXUkc7z77oBLaGUjEV"  # extracted from the URL
DRIVE_OUT = "gdrive_kitchenware.pdf"

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"  # lightweight, adjust if you prefer


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant connection
# ─────────────────────────────────────────────────────────────────────────────
def get_qdrant() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # Embedded/local store
    path = QDRANT_PATH or "./qdrant_local"
    os.makedirs(path, exist_ok=True)
    return QdrantClient(path=path)


# ─────────────────────────────────────────────────────────────────────────────
# Redis connection
# ─────────────────────────────────────────────────────────────────────────────
class _InMemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str):
        import time as _t
        v = self._store.get(key)
        if not v:
            return None
        val, exp = v
        if exp and exp < _t.time():
            self._store.pop(key, None)
            return None
        return val

    def setex(self, key: str, ttl_seconds: int, value: str):
        import time as _t
        self._store[key] = (value, _t.time() + max(1, int(ttl_seconds)))


def get_redis():
    """Return a cache-like object with get/setex. Uses Redis if available, else in-memory fallback."""
    if HAS_REDIS:
        try:
            return redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
        except Exception:
            pass
    return _InMemoryCache()


# ─────────────────────────────────────────────────────────────────────────────
# PDF reading & chunking
# ─────────────────────────────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    if not os.path.exists(path):
        return ""
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts).strip()


def token_len(enc, text: str) -> int:
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def chunk_text(
    text: str, max_tokens: int = 1000, overlap: int = 150, encoding_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    # Split by paragraphs then merge to token budget
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    curr = []
    curr_tokens = 0

    def flush():
        nonlocal curr, curr_tokens
        if curr:
            ct = "\n\n".join(curr)
            chunks.append(
                {"id": str(uuid.uuid4()), "text": ct, "n_tokens": token_len(enc, ct)}
            )
            curr = []
            curr_tokens = 0

    for p in paras:
        p_tokens = token_len(enc, p)
        if p_tokens > max_tokens:
            # hard split long para
            words = p.split()
            piece = []
            piece_tok = 0
            for w in words:
                wt = token_len(enc, w + " ")
                if piece_tok + wt > max_tokens:
                    chunks.append(
                        {"id": str(uuid.uuid4()), "text": " ".join(piece), "n_tokens": piece_tok}
                    )
                    # overlap by tokens ~ approximate words
                    if overlap > 0 and len(piece) > 0:
                        back = max(1, min(len(piece), overlap // 2))
                        piece = piece[-back:]
                        piece_tok = token_len(enc, " ".join(piece) + " ")
                    else:
                        piece = []
                        piece_tok = 0
                piece.append(w)
                piece_tok += wt
            if piece:
                chunks.append(
                    {"id": str(uuid.uuid4()), "text": " ".join(piece), "n_tokens": piece_tok}
                )
            continue

        if curr_tokens + p_tokens <= max_tokens:
            curr.append(p)
            curr_tokens += p_tokens
        else:
            flush()
            # start new with overlap from previous tail
            if overlap > 0 and chunks:
                tail = chunks[-1]["text"]
                tail_words = tail.split()
                back = max(1, min(len(tail_words), overlap // 2))
                curr = [" ".join(tail_words[-back:]), p]
                curr_tokens = token_len(enc, curr[0] + " " + p)
            else:
                curr = [p]
                curr_tokens = p_tokens

    flush()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding & upsert
# ─────────────────────────────────────────────────────────────────────────────
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        msg = str(e)
        if "invalid_api_key" in msg or "Incorrect API key" in msg or "401" in msg:
            st.error(
                "OpenAI authentication failed. Please set a valid OPENAI_API_KEY in your .env (no quotes, no spaces). "
                "You can create a key at https://platform.openai.com/account/api-keys. After updating, restart Streamlit."
            )
        raise


def ensure_collection(qc: QdrantClient, dim: int = 3072):
    # text-embedding-3-large output is 3072-d
    collections = [c.name for c in qc.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        qc.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def collection_is_empty(qc: QdrantClient) -> bool:
    try:
        stats = qc.get_collection(QDRANT_COLLECTION)
        return (stats.points_count or 0) == 0
    except Exception:
        return True


def upsert_chunks(qc: QdrantClient, chunks: List[Dict[str, Any]], source: str):
    if not chunks:
        return
    vectors = embed_texts([c["text"] for c in chunks])
    points = []
    for c, v in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=c["id"],
                vector=v,
                payload={"text": c["text"], "source": source, "n_tokens": c["n_tokens"]},
            )
        )
    qc.upsert(collection_name=QDRANT_COLLECTION, points=points)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def search_chunks(qc: QdrantClient, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    qvec = embed_texts([query])[0]
    res = qc.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
    )
    docs = []
    for r in res:
        payload = r.payload or {}
        docs.append({"text": payload.get("text", ""), "source": payload.get("source", ""), "score": r.score})
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Drive download
# ─────────────────────────────────────────────────────────────────────────────
def download_from_drive(file_id: str, out_path: str) -> bool:
    try:
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=True)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state & nodes
# ─────────────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    use_cache: bool
    cached_answer: Optional[str]
    docs: List[Dict[str, Any]]
    answer: Optional[str]


@dataclass
class Tools:
    qdrant: QdrantClient
    rds: Any


def node_cache_lookup(state: GraphState, tools: Tools) -> GraphState:
    q = state["question"].strip()
    ans = None
    if state.get("use_cache", True):
        try:
            key = f"grocery_cache::{q.lower()}"
            ans = tools.rds.get(key)
        except Exception:
            ans = None
    state["cached_answer"] = ans
    return state


def node_retrieve(state: GraphState, tools: Tools) -> GraphState:
    if state.get("cached_answer"):
        state["docs"] = []
        return state
    docs = search_chunks(tools.qdrant, state["question"], top_k=6)
    state["docs"] = docs
    return state


def node_generate(state: GraphState, tools: Tools) -> GraphState:
    if state.get("cached_answer"):
        state["answer"] = state["cached_answer"]
        return state

    sys_prompt = (
        "You are a helpful grocery store assistant. Answer briefly, cite product details "
        "from the provided context. If you are unsure, say so. Use bullet points for lists."
    )
    context = ""
    for i, d in enumerate(state.get("docs", []), 1):
        context += f"\n[Doc {i} | {d.get('source','')}] {d['text'][:1200]}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    state["answer"] = resp.choices[0].message.content.strip()
    return state


def node_cache_store(state: GraphState, tools: Tools) -> GraphState:
    if state.get("answer") and not state.get("cached_answer"):
        try:
            key = f"grocery_cache::{state['question'].strip().lower()}"
            tools.rds.setex(key, 60 * 60 * 24 * 7, state["answer"])  # 7d TTL
        except Exception:
            pass
    return state


def build_graph(tools: Tools):
    g = StateGraph(GraphState)
    g.add_node("cache_lookup", lambda s: node_cache_lookup(s, tools))
    g.add_node("retrieve", lambda s: node_retrieve(s, tools))
    g.add_node("generate", lambda s: node_generate(s, tools))
    g.add_node("cache_store", lambda s: node_cache_store(s, tools))

    g.set_entry_point("cache_lookup")
    g.add_edge("cache_lookup", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "cache_store")
    g.add_edge("cache_store", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# One-time ingestion
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_ingestion() -> Dict[str, Any]:
    qclient = get_qdrant()
    ensure_collection(qclient)
    if not collection_is_empty(qclient):
        return {"status": "exists", "ingested": 0}

    total_chunks = 0

    # Local PDF
    local_text = read_pdf(LOCAL_PDF)
    local_chunks = chunk_text(local_text, max_tokens=1000, overlap=150)
    if local_chunks:
        upsert_chunks(qclient, local_chunks, source=f"local:{os.path.basename(LOCAL_PDF)}")
        total_chunks += len(local_chunks)

    # Drive PDF
    ok = download_from_drive(DRIVE_ID, DRIVE_OUT)
    if ok:
        drive_text = read_pdf(DRIVE_OUT)
        drive_chunks = chunk_text(drive_text, max_tokens=1000, overlap=150)
        if drive_chunks:
            upsert_chunks(qclient, drive_chunks, source="gdrive:mukti_kitchenware")
            total_chunks += len(drive_chunks)

    return {"status": "ingested", "ingested": total_chunks}


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Grocery Store Chatbot", page_icon="🛒", layout="wide")

st.title("🛒 Grocery Store Chatbot")
st.caption("LangGraph + Qdrant + Redis cache + OpenAI")

with st.expander("⚙️ Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Qdrant Collection: `{QDRANT_COLLECTION}`")
        st.write("Embedding model:", EMBED_MODEL)
        st.write("Chat model:", CHAT_MODEL)
    with col2:
        st.write("Local PDF:", os.path.basename(LOCAL_PDF))
        st.write("Drive File ID:", DRIVE_ID)
    with col3:
        if st.button("Re-ingest (force rebuild)"):
            # clear collection by recreating
            qc = get_qdrant()
            ensure_collection(qc)
            qc.delete_collection(QDRANT_COLLECTION)
            ensure_collection(qc)
            st.success("Collection cleared. It will re-ingest on next question.")

# Show ingestion status / trigger once
with st.status("Preparing vector store (first run may take a minute)...", expanded=False) as status:
    info = run_ingestion()
    if info["status"] == "exists":
        status.update(label="Vector store ready", state="complete")
    else:
        status.update(label=f"Ingested {info['ingested']} chunks", state="complete")

# Chat input
if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.text_input("Ask about kitchenware, availability, features, etc.", placeholder="e.g., Do you have non-stick pans for induction stoves?")
cache_toggle = st.checkbox("Use Redis cache", value=True)
ask = st.button("Ask")

# Render history
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)

if ask and user_q.strip():
    st.session_state.history.append(("user", user_q))
    # Build tools + graph
    qclient = get_qdrant()
    r = get_redis()
    tools = Tools(qdrant=qclient, rds=r)
    graph = build_graph(tools)

    init_state: GraphState = {
        "question": user_q.strip(),
        "use_cache": cache_toggle,
        "cached_answer": None,
        "docs": [],
        "answer": None,
    }

    with st.spinner("Thinking..."):
        out_state = graph.invoke(init_state)

    answer = out_state.get("answer") or "Sorry, I couldn't find that."
    st.session_state.history.append(("assistant", answer))
    st.rerun()
