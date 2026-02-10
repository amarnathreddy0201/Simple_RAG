from langchain_community.vectorstores import Chroma

# from langchain_huggingface import HuggingFaceEmbeddings

# VECTOR_DB_PATH = "./vector_db"


# class Retriever:
#     def __init__(self):
#         self.embedding = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.db = Chroma(
#             persist_directory=VECTOR_DB_PATH,
#             embedding_function=self.embedding
#         )

#     def retrieve(self, query, user_id, document_id=None, k=2):
#         filters = {"user_id": user_id}
#         if document_id:
#             filters["document_id"] = document_id

#         return self.db.similarity_search(
#             query,
#             k=k,
#             filter=filters
#         )

# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "./vector_db"


class Retriever:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embedding
        )

    def retrieve(self, query, user_id, document_id=None, k=2):
        # Build proper Chroma filter
        if document_id:
            filter_query = {
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id}
                ]
            }
        else:
            filter_query = {"user_id": user_id}

        return self.db.similarity_search(
            query,
            k=k,
            filter=filter_query
        )
