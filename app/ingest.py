"""Data ingestion script for Promtior website content."""

import shutil
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from app.config import (
    PROMTIOR_URL,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


def ingest_data():
    """Load, split, and store Promtior website content."""

    print(f"🌐 Loading data from {PROMTIOR_URL}...")
    loader = WebBaseLoader(PROMTIOR_URL)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} document(s)")

    print(f"\n📄 Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(splits)} chunks")

    # Clean up old database
    if CHROMA_DB_DIR.exists():
        print(f"\n🗑️  Removing old vector store...")
        shutil.rmtree(CHROMA_DB_DIR)

    CHROMA_DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🔮 Creating embeddings with Ollama ({OLLAMA_MODEL})...")
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    print(f"💾 Storing in ChromaDB at {CHROMA_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=str(CHROMA_DB_DIR)
    )

    print(f"\n✅ Ingestion complete! {len(splits)} chunks stored in vector database.")


if __name__ == "__main__":
    try:
        ingest_data()
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Install from https://ollama.com")
        print(f"  2. Run: ollama pull {OLLAMA_MODEL}")
        print("  3. Ollama should be running in the background")
        raise
