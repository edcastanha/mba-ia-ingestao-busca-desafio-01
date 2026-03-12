import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ingest import ingest_pdf  # noqa


@patch('ingest.os.path.exists')
@patch('builtins.print')
def test_ingest_pdf_file_not_found(mock_print, mock_exists):
    mock_exists.return_value = False

    ingest_pdf()

    mock_exists.assert_called_once()
    mock_print.assert_called_once_with(
        "Erro: Arquivo PDF não encontrado no caminho: document.pdf")


@patch('ingest.os.path.exists')
@patch('ingest.PyPDFLoader')
@patch('ingest.RecursiveCharacterTextSplitter')
@patch('ingest.get_vector_store')
@patch('builtins.print')
def test_ingest_pdf_success(mock_print, mock_get_vector_store, mock_text_splitter_class, mock_pdf_loader_class, mock_exists):
    mock_exists.return_value = True

    # Mocking Loader
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = ["Doc 1", "Doc 2"]
    mock_pdf_loader_class.return_value = mock_loader_instance

    # Mocking Splitter
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_documents.return_value = [
        "Chunk 1", "Chunk 2", "Chunk 3"]
    mock_text_splitter_class.return_value = mock_splitter_instance

    # Mocking PGVector
    mock_vector_store_instance = MagicMock()
    mock_get_vector_store.return_value = mock_vector_store_instance

    ingest_pdf()

    mock_pdf_loader_class.assert_called_once_with("document.pdf")
    mock_loader_instance.load.assert_called_once()
    mock_text_splitter_class.assert_called_once_with(
        chunk_size=1000, chunk_overlap=150)
    mock_splitter_instance.split_documents.assert_called_once_with([
                                                                   "Doc 1", "Doc 2"])

    mock_get_vector_store.assert_called_once()
    mock_vector_store_instance.add_documents.assert_called_once_with(
        ["Chunk 1", "Chunk 2", "Chunk 3"])

    assert mock_print.call_count == 5
