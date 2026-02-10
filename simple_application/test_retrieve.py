from retriever import Retriever

r = Retriever()
docs = r.retrieve("skills in resume")

for d in docs:
    print("----")
    print(d.page_content[:300])