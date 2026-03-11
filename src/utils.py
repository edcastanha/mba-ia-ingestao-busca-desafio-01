import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()


def get_embeddings():
    """Retorna a classe de embeddings baseada na variável de ambiente setada."""
    #if os.environ.get("OPENAI_API_KEY"):
    #    model = os.environ.get("OPENAI_EMBEDDING_MODEL",
    #                           "text-embedding-3-small")
    #    return OpenAIEmbeddings(model=model)
    if os.environ.get("GOOGLE_API_KEY"):
        model = os.environ.get("GOOGLE_EMBEDDING_MODEL",
                               "models/gemini-embedding-001")
        if model == "models/text-embedding-004":
            model = "models/gemini-embedding-001"
        return GoogleGenerativeAIEmbeddings(model=model)
    else:
        raise ValueError(
            "Nenhuma chave de API encontrada (OPENAI ou GOOGLE). Preencha o arquivo .env.")


def get_llm():
    """Retorna o modelo LLM baseado na variável de ambiente setada."""
    #if os.environ.get("OPENAI_API_KEY"):
    # Defaulting to a standard text gen model, gpt-4o-mini is best for simple queries
    #    return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if os.environ.get("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    else:
        raise ValueError(
            "Nenhuma chave de API encontrada (OPENAI ou GOOGLE). Preencha o arquivo .env.")


def get_vector_store():
    """Retorna a instância do PGVector conectada ao banco."""
    connection = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5434/rag")
    collection_name = os.environ.get(
        "PG_VECTOR_COLLECTION_NAME", "rag_collection")
    embeddings = get_embeddings()

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
