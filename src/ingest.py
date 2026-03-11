import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import get_vector_store

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Erro: Arquivo PDF não encontrado no caminho: {PDF_PATH}")
        return

    print(f"📄 Carregando arquivo PDF: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"✂️ Dividindo o documento em chunks de 1000 caracteres (overlap de 150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📊 Foram criados {len(chunks)} chunks.")

    print(f"💾 Conectando ao banco de dados e gerando embeddings...")
    vector_store = get_vector_store()

    # Store documents
    vector_store.add_documents(chunks)
    print("✅ Ingestão finalizada com sucesso. Os vetores foram armazenados no PostgreSQL!")


if __name__ == "__main__":
    ingest_pdf()
