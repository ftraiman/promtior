"""LangServe API server for the Promtior chatbot."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from app.chain import get_retrieval_chain

# Create FastAPI app
app = FastAPI(
    title="Promtior RAG Chatbot API",
    version="1.0.0",
    description="API para chatbot RAG sobre Promtior usando LangChain y Ollama",
)


@app.get("/")
async def redirect_root_to_docs():
    """Redirect root to playground."""
    return RedirectResponse("/promtior-bot/playground")


# Initialize chain
try:
    chain = get_retrieval_chain()

    # Add LangServe routes
    add_routes(
        app,
        chain,
        path="/promtior-bot",
        enabled_endpoints=["invoke", "stream", "stream_log", "playground"],
    )

    print("✅ Server initialized successfully!")
    print("📍 Endpoints:")
    print("   - Playground: http://localhost:8000/promtior-bot/playground")
    print("   - Invoke: http://localhost:8000/promtior-bot/invoke")
    print("   - Stream: http://localhost:8000/promtior-bot/stream")

except Exception as e:
    print(f"❌ Error initializing server: {e}")
    print("\nMake sure to:")
    print("  1. Run data ingestion first: python -m app.ingest")
    print("  2. Ensure Ollama is running")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
