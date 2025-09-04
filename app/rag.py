import os
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Any, Optional


def build_context(query: str, texts: List[str], embedder, k: int = 6) -> List[str]:
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    # Use a transient FAISS for QA search
    d = embedder.get_sentence_embedding_dimension()
    idx = faiss.IndexFlatIP(d)
    corpus_embs = embedder.encode(texts, normalize_embeddings=True).astype(np.float32)
    idx.add(corpus_embs)
    sims, ids = idx.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in ids[0]]


def openai_chat(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: simple extractive concat of top snippets
        return "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    try:
        
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error] {e}"
