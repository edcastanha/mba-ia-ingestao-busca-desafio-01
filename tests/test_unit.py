from search import search_context
from utils import get_embeddings, get_llm
import os
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


@patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}, clear=True)
@patch('utils.VertexAIEmbeddings')
def test_get_embeddings_google(mock_vertex_ai):
    embeddings = get_embeddings()
    mock_vertex_ai.assert_called_once()
    assert embeddings is not None


@patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}, clear=True)
@patch('utils.ChatVertexAI')
def test_get_llm_google(mock_chat_vertex_ai):
    llm = get_llm()
    mock_chat_vertex_ai.assert_called_once()
    assert llm is not None


@patch.dict(os.environ, {}, clear=True)
def test_get_embeddings_no_key():
    with pytest.raises(ValueError, match="Nenhuma chave de API encontrada ou credentials do gcloud."):
        get_embeddings()


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
