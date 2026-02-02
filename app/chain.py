"""RAG chain implementation using LangChain."""

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import CHROMA_DB_DIR, OLLAMA_MODEL, OLLAMA_BASE_URL, RETRIEVAL_K


def get_retrieval_chain():
    """Create and return the RAG retrieval chain."""

    # Check if vector store exists
    if not CHROMA_DB_DIR.exists():
        raise ValueError(
            f"Vector store not found at {CHROMA_DB_DIR}. "
            "Please run 'python -m app.ingest' first."
        )

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # Load vector store
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # Initialize LLM
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    # Create prompt template
    template = """Eres un asistente experto sobre Promtior. Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto.
Si no puedes responder con la información proporcionada, di "No tengo suficiente información para responder esa pregunta."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    # Test the chain
    chain = get_retrieval_chain()

    test_questions = [
        "¿Qué servicios ofrece Promtior?",
        "¿Cuándo fue fundada la empresa?",
    ]

    print("🧪 Testing RAG chain...\n")
    for question in test_questions:
        print(f"❓ {question}")
        response = chain.invoke(question)
        print(f"💬 {response}\n")
