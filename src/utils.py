import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_postgres import PGVector

# Suprime os avisos de depreciação do LangChain para manter o CLI limpo
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

load_dotenv()


def get_embeddings():
    """Retorna a classe de embeddings baseada na variável de ambiente setada."""
    # if os.environ.get("OPENAI_API_KEY"):
    #    model = os.environ.get("OPENAI_EMBEDDING_MODEL",
    #                           "text-embedding-3-small")
    #    return OpenAIEmbeddings(model=model)
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "mba-ia")
        return VertexAIEmbeddings(model_name=model, project=project)
    else:
        raise ValueError(
            "Nenhuma chave de API encontrada ou credentials do gcloud. Preencha o arquivo .env ou faça login gcloud.")


def get_llm():
    """Retorna o modelo LLM baseado na variável de ambiente setada."""
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "mba-ia")
        return ChatVertexAI(model="gemini-1.5-flash-002", temperature=0, project=project)
    else:
        raise ValueError(
            "Nenhuma chave de API encontrada ou credentials do gcloud. Preencha o arquivo .env ou faça login gcloud.")


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
