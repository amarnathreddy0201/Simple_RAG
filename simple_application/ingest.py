# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter




VECTOR_DB_PATH = "./vector_db"


def ingest_pdf(pdf_path, doc_id="default_doc"):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Add metadata
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = pdf_path

    # Create embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load or create vector DB
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding
    )

    # Add documents
    vector_db.add_documents(chunks)
    vector_db.persist()

    print("PDF ingested successfully!")


if __name__ == "__main__":
    ingest_pdf(r"D:\Family\Amar\Amar_resume.pdf", doc_id="sample")
