"""Configuration settings for the Promtior chatbot."""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Data source
PROMTIOR_URL = "https://promtior.ai/"

# Text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
RETRIEVAL_K = 4  # Number of documents to retrieve
