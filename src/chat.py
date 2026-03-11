from search import search_context
from utils import get_llm


def main():
    print("=========================================================")
    print(" IA Assistant do RAG - SuperTechIABrazil")
    print(" Digite sua pergunta sobre o balanço da empresa.")
    print(" Para encerrar o chat, digite 'sair', 'quit' ou 'exit'")
    print("=========================================================\n")

    try:
        llm = get_llm()
    except Exception as e:
        print(f"Não foi possível iniciar o chat. Erro de inicialização: {e}")
        return

    while True:
        try:
            pergunta_usuario = input("\nFaça sua pergunta: \n\nPERGUNTA: ")

            # Condição de saída
            if pergunta_usuario.strip().lower() in ['sair', 'quit', 'exit']:
                print("\nEncerrando o chat. Até logo!")
                break

            if not pergunta_usuario.strip():
                continue

            # 1. Recupera o contexto do PGVector e monta a string do Prompt
            prompt_montado = search_context(pergunta_usuario)

            # 2. Envia para o modelo LLM
            resposta = llm.invoke(prompt_montado)

            # 3. Exibe o resultado
            print(f"RESPOSTA: {resposta.content}")

        except KeyboardInterrupt:
            print("\nEncerrando o chat. Até logo!")
            break
        except Exception as e:
            print(f"\nErro ao processar sua pergunta: {e}")


if __name__ == "__main__":
    main()
