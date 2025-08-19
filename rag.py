import os, json, pathlib
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=str(pathlib.Path(__file__).parent / ".env"), override=True)

# -----------------------
# Parámetros principales
# -----------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "2"))  # menos contexto = más rápido en CPU
INDEX_DIR = pathlib.Path("storage")
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.json"

_model = None

def get_embedder():
    """Carga el modelo de embeddings solo desde cache local (sin red)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _model

def embed_texts(texts: List[str]):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def ensure_index_exists():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("No existe el índice. Ejecuta primero: python ingest.py")

def retrieve(query: str) -> List[Dict[str, Any]]:
    ensure_index_exists()
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    index = faiss.read_index(str(INDEX_FILE))
    qv = embed_texts([query])
    D, I = index.search(qv, TOP_K)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        item = meta["chunks"][idx]
        item["score"] = float(D[0][rank])
        results.append(item)
    return results

# -----------------------
# Generación
# -----------------------
def generate_answer(prompt: str) -> str:
    # Opción A: Ollama local
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        try:
            from ollama import Client
            host        = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            timeout     = float(os.getenv("OLLAMA_TIMEOUT", "600"))   # más margen
            num_predict = int(os.getenv("NUM_PREDICT", "60"))         # menos tokens = más rápido
            temperature = float(os.getenv("RAG_TEMPERATURE", "0.2"))
            num_thread  = int(os.getenv("NUM_THREAD", "8"))           # tu CPU tiene 8 hilos lógicos
            num_ctx     = int(os.getenv("NUM_CTX", "1024"))           # contexto corto reduce cómputo
            keep_alive  = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

            client = Client(host=host, timeout=timeout)
            resp = client.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": num_predict,
                    "temperature": temperature,
                    "num_thread": num_thread,
                    "num_ctx": num_ctx,
                },
                keep_alive=keep_alive,
                stream=False,
            )
            return resp["message"]["content"]
        except Exception as e:
            return f"[ERROR OLLAMA] {e}"
        
    # Opción B: Anthropic (Claude)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
            client = Anthropic()
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
                messages=[{"role":"user","content": prompt}],
            )
            # Claude devuelve una lista de "content" (bloques). Tomamos el texto.
            return "".join(part.text for part in resp.content if getattr(part, "text", None))
        except Exception as e:
            return f"[ERROR CLAUDE] {e}"

    # Opción C: OpenAI
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Sos un asistente conciso y preciso. Responde en español y cita las fuentes provistas cuando corresponda."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR OPENAI] {e}"

def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        [f"[{i+1}] (archivo: {c['source']})\n{c['text']}" for i, c in enumerate(contexts)]
    )
    return f"""Usando exclusivamente la siguiente información de contexto (fragmentos recuperados del CV), respondé en español de manera directa.
Si no está en los textos, decí honestamente que no aparece en el CV.
Respondé en ≤ 3 bullets y máx. 60 palabras.

Pregunta: {query}

Contexto:
{context_block}

Instrucciones:
- Integrá y sintetizá la información relevante.
- Al final, añadí una sección "Fuentes" con los índices de fragmentos usados (ej: [1], [3]) y el nombre de archivo.
"""

def rag_answer(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    answer = generate_answer(prompt)
    return answer, contexts
