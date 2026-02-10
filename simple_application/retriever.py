

# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings


# VECTOR_DB_PATH = "./vector_db"


# class Retriever:
#     def __init__(self):
#         embedding = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.vector_db = Chroma(
#             persist_directory=VECTOR_DB_PATH,
#             embedding_function=embedding
#         )

#     def get_relevant_docs(self, query, k=4):
#         return self.vector_db.similarity_search(query, k=k)

# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "./vector_db"


class Retriever:
    def __init__(self):
        # Same embedding model used in ingest.py
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load persistent Chroma DB
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embedding
        )

    def retrieve(self, query, k=10):
        """
        Retrieve top-k relevant chunks
        """
        docs = self.vector_db.similarity_search(query, k=k)
        return docs
