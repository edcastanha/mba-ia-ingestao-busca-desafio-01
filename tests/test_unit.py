import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from search import search_context  # noqa
from utils import get_embeddings, get_llm  # noqa


@patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}, clear=True)
@patch('utils.GoogleGenerativeAIEmbeddings')
def test_get_embeddings_google(mock_google_ai):
    embeddings = get_embeddings()
    mock_google_ai.assert_called_once()
    assert embeddings is not None


@patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}, clear=True)
@patch('utils.ChatGoogleGenerativeAI')
def test_get_llm_google(mock_chat_google_ai):
    llm = get_llm()
    mock_chat_google_ai.assert_called_once()
    assert llm is not None


@patch.dict(os.environ, {}, clear=True)
def test_get_embeddings_no_key():
    with pytest.raises(
            ValueError,
            match=r"Nenhuma chave de API \(OPENAI_API_KEY, GOOGLE_API_KEY\) ou OLLAMA_MODEL foi encontrada\."
    ):
        get_embeddings()


@patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"}, clear=True)
@patch('utils.OpenAIEmbeddings')
def test_get_embeddings_openai(mock_openai):
    embeddings = get_embeddings()
    mock_openai.assert_called_once()
    assert embeddings is not None


@patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"}, clear=True)
@patch('utils.ChatOpenAI')
def test_get_llm_openai(mock_chat_openai):
    llm = get_llm()
    mock_chat_openai.assert_called_once()
    assert llm is not None


@patch.dict(os.environ, {"OLLAMA_MODEL": "llama-fake"}, clear=True)
@patch('utils.OllamaEmbeddings')
def test_get_embeddings_ollama(mock_ollama):
    embeddings = get_embeddings()
    mock_ollama.assert_called_once()
    assert embeddings is not None


@patch.dict(os.environ, {"OLLAMA_MODEL": "llama-fake"}, clear=True)
@patch('utils.ChatOllama')
def test_get_llm_ollama(mock_chat_ollama):
    llm = get_llm()
    mock_chat_ollama.assert_called_once()
    assert llm is not None


@patch('search.get_vector_store')
def test_search_context(mock_get_vector_store):
    # Mock Document
    class MockDoc:
        def __init__(self, page_content):
            self.page_content = page_content

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = [
        (MockDoc("Fake context chunk 1"), 0.9),
        (MockDoc("Fake context chunk 2"), 0.8)
    ]
    mock_get_vector_store.return_value = mock_vector_store

    prompt = search_context("Qual o faturamento?")

    assert "Fake context chunk 1" in prompt
    assert "Fake context chunk 2" in prompt
    assert "Qual o faturamento?" in prompt
