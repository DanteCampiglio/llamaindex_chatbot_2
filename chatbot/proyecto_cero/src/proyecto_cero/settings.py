from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Paths
    project_root: Path
    raw_dir: Path
    interim_dir: Path
    index_dir: Path

    # RAG params
    embeddings_provider: str = "sentence-transformers"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embeddings_model: str = "text-embedding-3-small"
    collection_name: str = "pdf_chunks"
    chunk_chars: int = 2000
    overlap_chars: int = 200
    top_k: int = 6

    # LLM params
    # Default to xAI Grok as requested; can be overridden via .env LLM_PROVIDER
    llm_provider: str = "xai"
    groq_model: str = "llama3-8b-8192"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1:8b"
    bedrock_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[2]

        # Load .env for API keys only
        try:
            from dotenv import load_dotenv
            dotenv_path = root / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path=dotenv_path)
        except Exception:
            pass

        # Use class defaults for non-sensitive settings
        return cls(
            project_root=root,
            raw_dir=root / "data" / "raw",
            interim_dir=root / "data" / "interim", 
            index_dir=root / "data" / "index",
            # All other params will use their default values
        )
