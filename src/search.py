from utils import get_vector_store
from langchain_core.prompts import PromptTemplate
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_context(query: str):
    """Realiza uma busca isolada no PGVector e retorna os textos concatenados e o prompt"""
    vector_store = get_vector_store()

    # k=10 from the requirements
    results = vector_store.similarity_search_with_score(query, k=10)

    # Concatenate the page_contents from the results
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(contexto=context_text, pergunta=query)

    return formatted_prompt


if __name__ == "__main__":
    print("Testador de Busca Vetorial Semântica")
    query = input("Digite sua busca: ")
    if query.strip():
        print("Buscando e montando prompt...")
        prompt_result = search_context(query)
        print("\n--- PROMPT MONTADO ---\n")
        print(prompt_result)
