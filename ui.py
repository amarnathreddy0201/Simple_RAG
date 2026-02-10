import streamlit as st
import os
from ingest import ingest_pdf
from retriever import Retriever
from llm import LocalLLM

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Multi-User RAG", layout="centered")
st.title("📄 Multi-User Document Chatbot (RAG)")

# ---------------- USER ----------------
user_id = st.text_input("👤 User ID")

# ---------------- UPLOAD ----------------
st.subheader("Upload Document")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
document_id = st.text_input("Document ID (unique per user)")

if st.button("Upload & Save"):
    if user_id and uploaded_file and document_id:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ingest_pdf(file_path, user_id, document_id)
        st.success("✅ Document saved successfully")
    else:
        st.error("Please fill all fields")

# ---------------- CHAT ----------------
st.divider()
st.subheader("Ask Questions")

query = st.text_input("Your question")

@st.cache_resource
def load_retriever():
    return Retriever()

@st.cache_resource
def load_llm():
    return LocalLLM()

if st.button("Ask"):
    if user_id and query:
        retriever = load_retriever()
        llm = load_llm()

        docs = retriever.retrieve(query, user_id, document_id or None)
        context = "\n\n".join([d.page_content for d in docs])
        context = context[:1500]  # speed control

        prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        answer = llm.generate(prompt)
        st.write("### Answer")
        st.write(answer)
    else:
        st.error("Missing user ID or question")
