import os
from typing import Any

from langchain_ollama import OllamaEmbeddings


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return (v.strip() if isinstance(v, str) and v.strip() else default)


# Ollama 設定（可用環境變數覆寫）
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_embedding_model() -> Any:
    """
    目前先提供最小可用的 embedding：OllamaEmbeddings。
    後續若要擴充 Gemini / 其他後端，可用 RAG_EMBEDDING_BACKEND 做分流。
    """
    backend = _env("RAG_EMBEDDING_BACKEND", "ollama").lower()
    if backend != "ollama":
        raise ValueError(f"Unsupported RAG_EMBEDDING_BACKEND={backend!r}. Currently only 'ollama' is supported.")
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

