from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable
from langchain.chains import RetrievalQA    

from openai import OpenAI
from typing import Optional, List
from pydantic import Field
import os


# Set Together API Key
os.environ["OPENAI_API_KEY"] = "37023c695b45a31148940bc754ea0cf6199373b64c5816987c9d8239939e314e"
together_client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["OPENAI_API_KEY"]
)

# LLM Wrapper
# LLM Wrapper using Together.ai - Gemma 3 1B IT
class TogetherLLM(LLM, Runnable):
    model_name: str = "google/gemma-1.1-1b-it"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        response = together_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stop=stop or ["</s>"],
        )
        return response.choices[0].message.content.strip()

    def invoke(self, input: str, **kwargs) -> str:
        return self._call(input, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "together-ai"


# Step 1: Load your personal info
loader = TextLoader("myinfo.txt", encoding="utf-8")
documents = loader.load()

# Step 2: Embed the documents using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding_model)
retriever = db.as_retriever()
llm = TogetherLLM()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = qa.invoke(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


