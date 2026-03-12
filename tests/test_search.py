import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from search import search_context  # noqa


@patch('search.get_vector_store')
@patch('builtins.input', side_effect=['Qual é a receita?', ''])
def test_search_main_interactive(mock_input, mock_get_vector_store):
    # Mock Document format for PGVector results
    class MockDoc:
        def __init__(self, page_content):
            self.page_content = page_content

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = [
        (MockDoc("O faturamento em 2024 foi ótimo"), 0.9)
    ]
    mock_get_vector_store.return_value = mock_vector_store

    import search

    with patch('sys.stdout', new=StringIO()) as fake_out:
        # Recreating the main block logic manually because it only runs under __name__ == '__main__'
        print("Testador de Busca Vetorial Semântica")
        query = mock_input()
        if query.strip():
            print("Buscando e montando prompt...")
            prompt_result = search.search_context(query)
            print("\n--- PROMPT MONTADO ---\n")
            print(prompt_result)

    output = fake_out.getvalue()

    # Asserting successful run output
    assert "Testador de Busca Vetorial Semântica" in output
    assert "Buscando e montando prompt..." in output
    assert "--- PROMPT MONTADO ---" in output
    assert "O faturamento em 2024 foi ótimo" in output
    assert "Qual é a receita?" in output
