import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from chat import main  # noqa


@patch('chat.get_llm')
@patch('chat.search_context')
@patch('builtins.input', side_effect=['Qual é a capital da França?', 'sair'])
def test_e2e_fake_out_of_context(mock_input, mock_search, mock_get_llm, capsys):
    # Mock dependencies
    mock_llm_instance = MagicMock()

    # LLM simulates what a restrictive prompt would force it to do
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Não tenho informações necessárias para responder sua pergunta."
    mock_llm_instance.invoke.return_value = mock_llm_response
    mock_get_llm.return_value = mock_llm_instance

    mock_search.return_value = "Prompt formatado com regras de restrição..."

    # Executa a função main do chat
    main()

    # Capture the output
    captured = capsys.readouterr()

    # Asserts
    assert mock_search.called
    assert mock_llm_instance.invoke.called
    assert "RESPOSTA: Não tenho informações necessárias para responder sua pergunta." in captured.out
    assert "Encerrando o chat. Até logo!" in captured.out
