# Desafio RAG Básico com LangChain e pgVector

Este projeto é uma aplicação de linha de comando (CLI) que realiza Ingestão de PDF em um banco de dados vetorial (PostgreSQL + pgVector) e permite realizar buscas semânticas exclusivas no conteúdo do arquivo inserido (*Retrieval-Augmented Generation*).

## Como Executar e Iniciar (Guia Rápido)

1. Instale as dependências (recomendado usar o `uv`):
   ```bash
   uv pip install -r requirements.txt
   ```
2. Configure o arquivo `.env` (copie do `.env.example`) definindo suas chaves de API (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`).
3. Inicialize o banco PostgreSQL:
   ```bash
   docker compose up -d
   ```
4. Processe o arquivo `document.pdf` para o banco de dados:
   ```bash
   uv run python src/ingest.py
   ```
5. Inicie sua sessão de Chat P&R:
   ```bash
   uv run python src/chat.py
   ```

*Nota: Estamos testando LLMs locais (Ollama com suporte a GPU) na branch `dev-ollama`. Confira para rodar inferências offline!*

---

## Documentação Oficial (MkDocs)

Nós aplicamos o **MkDocs** (com o tema *Material*) para garantir uma documentação contínua, rica e fácil de navegar. O MkDocs foi adotado para gerar e demonstrar manuais e estruturas extensas com busca semântica embutida.

Para demonstrar o uso do MkDocs e ler o manual detalhado com as arquiteturas do projeto:

1. **Instale as dependências da documentação:**
   ```bash
   uv pip install mkdocs mkdocs-material
   ```

2. **Inicie o servidor embutido do MkDocs:**
   ```bash
   mkdocs serve -a 127.0.0.1:8001
   ```

3. **Acesse no seu navegador:**
   [http://127.0.0.1:8001/](http://127.0.0.1:8001/)

> Ao iniciar `mkdocs serve`, os arquivos do diretório `docs/` são gerados "on-the-fly" convertendo Markdown em um layout elegante via navegador. Teste você também!