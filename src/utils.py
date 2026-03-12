import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_postgres import PGVector

# Suprime os avisos de depreciação do LangChain para manter o CLI limpo
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

load_dotenv()


def get_embeddings():
    """Retorna a classe de embeddings baseada na env var setada."""
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("OPENAI_EMBEDDING_MODEL",
                               "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    elif os.environ.get("GOOGLE_API_KEY"):
        model = os.environ.get("GOOGLE_EMBEDDING_MODEL",
                               "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model)
    elif os.environ.get("OLLAMA_MODEL"):
        model = os.environ.get("OLLAMA_MODEL")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model, base_url=base_url)
    else:
        raise ValueError(
            "Nenhuma chave de API (OPENAI_API_KEY, GOOGLE_API_KEY) ou "
            "OLLAMA_MODEL foi encontrada. Preencha o arquivo .env "
            "adequadamente."
        )


def get_llm():
    """Retorna o modelo LLM baseado na variável de ambiente setada."""
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    elif os.environ.get("GOOGLE_API_KEY"):
        model = os.environ.get("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    elif os.environ.get("OLLAMA_MODEL"):
        model = os.environ.get("OLLAMA_MODEL")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, temperature=0, base_url=base_url)
    else:
        raise ValueError(
            "Nenhuma chave de API (OPENAI_API_KEY, GOOGLE_API_KEY) ou "
            "OLLAMA_MODEL foi encontrada. Preencha o arquivo .env "
            "adequadamente."
        )


def get_vector_store():
    """Retorna a instância do PGVector conectada ao banco."""
    connection = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5434/rag"
    )
    collection_name = os.environ.get(
        "PG_VECTOR_COLLECTION_NAME", "rag_collection")
    embeddings = get_embeddings()

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
