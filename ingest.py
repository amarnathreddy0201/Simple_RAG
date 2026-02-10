import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "./vector_db"


def ingest_pdf(pdf_path, user_id, document_id):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["user_id"] = user_id
        chunk.metadata["document_id"] = document_id
        chunk.metadata["file_name"] = os.path.basename(pdf_path)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding
    )

    db.add_documents(chunks)
    db.persist()
