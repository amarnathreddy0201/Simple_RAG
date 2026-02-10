from retriever import Retriever
from llm import LocalLLM


class RAGChatbot:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LocalLLM()

    def answer(self, query):
        docs = self.retriever.retrieve(query)
        print("######")
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant.
Answer using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""
        
        response = self.llm.generate(prompt)
        return response


if __name__ == "__main__":
    bot = RAGChatbot()

    while True:
        query = input("\nWhat is this document? How many years of experience: ")
        if query.lower() == "exit":
            break

        answer = bot.answer(query)
        print("\nAnswer:\n", answer)
