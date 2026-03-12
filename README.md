# Desafio RAG Básico com LangChain e pgVector

Este projeto é uma aplicação de linha de comando (CLI) que realiza Ingestão de PDF em um banco de dados vetorial (PostgreSQL + pgVector) e permite realizar buscas semânticas (Retrieval-Augmented Generation) com base exclusiva no conteúdo do arquivo inserido.

## Tecnologias Utilizadas
* **Linguagem:** Python 3.12+ (gerenciado via `uv`)
* **Framework:** LangChain
* **Banco de Dados:** PostgreSQL com extensão pgVector
* **Modelos:** Suporte nativo para OpenAI e Google Gemini
* **Infraestrutura:** Docker e Docker Compose

---

## Estrutura do Projeto

```text
├── docker-compose.yml    # Sobe o banco PostgreSQL com pgVector
├── requirements.txt      # Dependências exportadas do projeto
├── pyproject.toml        # Configurações do projeto gerenciadas via uv
├── .env.example          # Template para as chaves de API
├── src/
│   ├── ingest.py         # Script de ingestão do PDF (divisão e vetorização)
│   ├── search.py         # Script autônomo de busca semântica (k=10)
│   ├── chat.py           # CLI para interação de P&R restrita
├── document.pdf          # PDF que servirá como base de conhecimento
└── README.md             # Instruções de execução
```

---

## Ordem de Execução e Setup

### 1. Criar e Ativar Ambiente Virtual
Recomendamos o uso do `uv` (já inicializado no projeto) ou `venv` padrão:

Usando `venv` clássico:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Usando `uv` (rápido):
```bash
uv pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente
Copie o arquivo de exemplo do `.env` e insira sua chave da OpenAI ou Google:
```bash
cp .env.example .env
```
Edite `.env` e defina apenas a chave que pretende usar (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`).
*A string de conexão do banco `DATABASE_URL` já está configurada por padrão na porta 5432.*

### 3. Subir o Banco de Dados
```bash
docker compose up -d
```
Aguarde alguns segundos para o banco iniciar e criar os contêineres corretamente.

### 4. Executar a Ingestão do Arquivo PDF
Certifique-se de que existe um arquivo `document.pdf` na raiz do projeto.
```bash
python src/ingest.py
```
*(Ou, se usar `uv` diretamente: `uv run python src/ingest.py`)*

### 5. Iniciar o Chat via CLI
```bash
python src/chat.py
```
*(Ou, se usar `uv` diretamente: `uv run python src/chat.py`)*

Faça suas perguntas baseadas no texto que você enviou.
Se a pergunta não constar no arquivo PDF, a aplicação é instruída a retornar estritamente a mensagem:
> **"Não tenho informações necessárias para responder sua pergunta."**

Digite `sair` para finalizar o chat.

---

## 🚀 Explorando LLMs Locais (Branch `dev-ollama`)

Nós estamos testando o uso de modelos locais através do **Ollama** com suporte a **GPU local** na branch `dev-ollama`.
Isso permite executar a aplicação RAG (ingestão, vetorização e chat) de forma totalmente offline e privada, aproveitando a aceleração de hardware local para inferência em vez de depender de APIs através da nuvem (como OpenAI ou Google Gemini).

Para conferir e testar o uso com Ollama:
```bash
git checkout dev-ollama
```

> **Nota:** Consulte as instruções do README na própria branch `dev-ollama` para mais detalhes de como proceder com a instalação do Ollama, download do modelo local e execução.