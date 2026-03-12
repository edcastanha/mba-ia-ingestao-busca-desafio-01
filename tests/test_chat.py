import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from chat import main  # noqa


@patch('chat.get_llm')
@patch('builtins.print')
def test_main_llm_init_error(mock_print, mock_get_llm):
    mock_get_llm.side_effect = Exception("Mock LLM Init Error")

    main()

    mock_print.assert_any_call(
        "Não foi possível iniciar o chat. Erro de inicialização: Mock LLM Init Error")


@patch('chat.get_llm')
@patch('builtins.input', side_effect=['', 'sair'])
def test_main_empty_input(mock_input, mock_get_llm):
    mock_llm_instance = MagicMock()
    mock_get_llm.return_value = mock_llm_instance

    with patch('sys.stdout', new=StringIO()) as fake_out:
        main()

    output = fake_out.getvalue()
    assert "Encerrando o chat. Até logo!" in output
    mock_llm_instance.invoke.assert_not_called()


@patch('chat.get_llm')
@patch('builtins.input', side_effect=KeyboardInterrupt)
def test_main_keyboard_interrupt(mock_input, mock_get_llm):
    mock_llm_instance = MagicMock()
    mock_get_llm.return_value = mock_llm_instance

    with patch('sys.stdout', new=StringIO()) as fake_out:
        main()

    output = fake_out.getvalue()
    assert "Encerrando o chat. Até logo!" in output


@patch('chat.get_llm')
@patch('chat.search_context')
@patch('builtins.input', side_effect=['Qual é a capital da França?', 'sair'])
def test_main_search_error(mock_input, mock_search, mock_get_llm):
    mock_llm_instance = MagicMock()
    mock_get_llm.return_value = mock_llm_instance

    # Simulate an error during context search
    mock_search.side_effect = Exception("Mock DB Connection Error")

    with patch('sys.stdout', new=StringIO()) as fake_out:
        main()

    output = fake_out.getvalue()
    assert "Erro ao processar sua pergunta: Mock DB Connection Error" in output
    assert "Encerrando o chat. Até logo!" in output
