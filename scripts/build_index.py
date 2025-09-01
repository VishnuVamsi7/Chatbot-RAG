from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

def load_documents():
    with open("data/myinfo.txt", "r", encoding="utf-8") as f:
        content = f.read().strip()
    return [Document(page_content=content)]

def build_and_save_index(docs, index_path="data/faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved to {index_path}")

if __name__ == "__main__":
    docs = load_documents()
    build_and_save_index(docs)
