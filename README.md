# Chatbot RAG para consultar el CV/alumno (Streamlit)

Sistema de **Retrieval-Augmented Generation (RAG)** que permite consultar un CV y obtener respuestas con citas de las secciones relevantes.
Listo para ejecutar localmente con **Streamlit + FAISS + Sentence-Transformers** y generación vía **Ollama** (local) u **OpenAI**/ **Claude** (SaaS).

## 🚀 Demo rápida (5 pasos)

```bash
# 1) Clonar o copiar esta carpeta
cd rag_cv_chatbot

# 2) Crear entorno e instalar deps
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
pip install -r requirements.txt

# 3) (Opcional) Configurar modelo de generación
#    Opción A - Local con Ollama (recomendado, gratis):
#      - Instalar https://ollama.com/download
#      - Descargar modelo:  ollama pull llama3.1:8b
#      - Exportar variable:  set OLLAMA_MODEL=llama3.1:8b   (Windows)
#                            export OLLAMA_MODEL=llama3.1:8b (Mac/Linux)#
#    Opción B - Claude:
#      - Copiar .env.example a .env y setear CLAUDE_API_KEY=<tu_api_key>
#    Opción C - OpenAI:
#      - Copiar .env.example a .env y setear OPENAI_API_KEY=<tu_api_key>

# 4) Colocar CV en /data como PDF/TXT/MD (ej: cv_alumno.pdf).
#    También podés subirlo desde la UI.
python ingest.py  # crea/actualiza el índice

# 5) Lanzar la app
streamlit run app.py
```

Abrir el navegador en la URL que imprime Streamlit (usualmente `http://localhost:8501`).

---

## 📁 Estructura

```
rag_cv_chatbot/
├─ app.py                # UI Streamlit (chat + upload + reindex)
├─ ingest.py             # Ingesta de documentos de /data -> índice FAISS
├─ rag.py                # Núcleo: embedder, retriever y generación
├─ utils.py              # Carga y troceo de documentos
├─ requirements.txt
├─ .env.example
├─ data/
│  └─ CV_ejemplo.md
├─ storage/              # Se crea al ejecutar ingest.py (índice y metadatos)
└─ docs/
```

## ✨ Características

- **Ingesta simple**: arrastrá tu CV (PDF/TXT/MD) o reemplazá el ejemplo y ejecutá `python ingest.py`.
- **Embeddings multilingües**: `paraphrase-multilingual-MiniLM-L12-v2` (soporta español).
- **FAISS** como vector store en disco.
- **RAG** con *top-k retrieval* y citas (fragmentos + archivo origen).
- **Modelo de texto** intercambiable: **Ollama local** o **OpenAI** (selección automática por variables de entorno).
- **Streamlit** con historial de chat y subida de archivos.

## 🧠 ¿Cómo funciona?

1. **Troceo del CV** en fragmentos (~500 tokens con solapamiento).
2. **Embeddings** de cada chunk y construcción del índice **FAISS**.
3. En consulta, se recuperan los **k** fragmentos más similares.
4. Se genera una respuesta con un **prompt con contexto** que incluye los fragmentos recuperados.
5. Se muestran **citas** (archivo y preview del texto) para trazabilidad.

## ⚙️ Configuración

- Variables de entorno (en `.env`):
  - `CLAUDE_API_KEY`: clave para usar modelos de Claude-code: `claude-3-7-sonnet-20250219`
  - `OPENAI_MODEL` (opcional): por defecto `gpt-4o-mini`.
  - `OLLAMA_MODEL` (opcional): p.ej. `llama3.1:8b` si usás Ollama local.

- Parámetros principales (editables en `rag.py`):
  - `EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
  - `CHUNK_SIZE = 1000` (caracteres aproximados)
  - `CHUNK_OVERLAP = 150`
  - `TOP_K = 4`

## 🧪 Pruebas rápidas de preguntas

- *"¿Cuál es la formación académica del alumno?"*
- *"Enumera experiencias relevantes en QA/IA con fechas."*
- *"¿Qué habilidades técnicas tiene y en qué proyectos las aplicó?"*
- *"¿Datos de contacto?"*

## 🧾 Licencia

MIT. 
