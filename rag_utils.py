import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

RAG_DIR = Path(__file__).resolve().parent
KB_PATH = RAG_DIR / "knowledge_base.md"
CHROMA_PATH = RAG_DIR / "chroma_db"
COLLECTION_NAME = "bot_knowledge"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_client = None
_collection = None
_embedding_fn = None


def _get_embedding_model():
    global _embedding_fn
    if _embedding_fn is None:
        from sentence_transformers import SentenceTransformer
        _embedding_fn = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_fn


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:

    blocks = re.split(r"\n+", text.strip())
    chunks = []
    current = []
    current_len = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_len = len(block) + 1
        if current_len + block_len > chunk_size and current:
            chunks.append("\n".join(current))
          
            overlap_len = 0
            new_current = []
            for s in reversed(current):
                overlap_len += len(s) + 1
                new_current.append(s)
                if overlap_len >= overlap:
                    break
            current = list(reversed(new_current))
            current_len = sum(len(s) + 1 for s in current)
        current.append(block)
        current_len += block_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def _load_documents() -> list[str]:
  
    texts = []
    if KB_PATH.exists():
        texts.append(KB_PATH.read_text(encoding="utf-8"))
    docs_dir = RAG_DIR / "docs"
    if docs_dir.exists():
        for p in docs_dir.glob("**/*.md"):
            texts.append(p.read_text(encoding="utf-8"))
        for p in docs_dir.glob("**/*.txt"):
            texts.append(p.read_text(encoding="utf-8"))
    if not texts:
        return []
    all_chunks = []
    for t in texts:
        all_chunks.extend(_chunk_text(t))
    return all_chunks


def build_index():
  
    global _client, _collection
    chunks = _load_documents()
    if not chunks:
        raise FileNotFoundError(f"База знаний не найдена: {KB_PATH} или папка docs/")

    model = _get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Knowledge for RAG"},
    )
    _collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )
    return len(chunks)


def _ensure_collection():
  
    global _client, _collection
    if _collection is not None:
        return
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    try:
        _collection = _client.get_collection(COLLECTION_NAME)
    except Exception:
        
        build_index()


def retrieve(question: str, top_k: int = 4) -> list[str]:

    _ensure_collection()
    model = _get_embedding_model()
    q_emb = model.encode([question], show_progress_bar=False).tolist()
    results = _collection.query(query_embeddings=q_emb, n_results=min(top_k, _collection.count()))
    docs = results["documents"][0] if results["documents"] else []
    return docs


def answer_with_llm(question: str, context_chunks: list[str]) -> str:
   
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
     
        return (
            "Ответ по базе знаний (без LLM, задайте OPENAI_API_KEY для полного RAG):\n\n"
            + "\n\n---\n\n".join(context_chunks[:3])
        )
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        context = "\n\n".join(context_chunks)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты помощник по бухгалтерии и расчёту зарплаты. Отвечай кратко и по существу только на основе приведённого контекста. Если в контексте нет ответа — так и скажи. Язык ответа: русский.",
                },
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос: {question}",
                },
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Ошибка при запросе к LLM: {e}. Контекст:\n\n" + "\n\n".join(context_chunks[:2])


def ask(question: str, top_k: int = 4) -> str:
 
    question = question.strip()
    if not question:
        return "Напишите вопрос текстом (например: как считаются отпускные?)."
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return "База знаний пуста. Добавьте knowledge_base.md или файлы в папку docs/."
    return answer_with_llm(question, chunks)

