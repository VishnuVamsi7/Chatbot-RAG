from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

from huggingface_hub import InferenceClient
from openai import OpenAI
from typing import Optional, List
import os
import dotenv
dotenv.load_dotenv()

# Initialize Together.ai client
hf_token = os.getenv("HF_API_KEY")
together_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token
)

# LLM Wrapper using Together.ai
class TogetherLLM(LLM, Runnable):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

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

# Hosted embedding class using Hugging Face Inference API
class HFHostedEmbeddings(Embeddings):
    def __init__(self, model_name="intfloat/e5-small-v2"):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_API_KEY")
        )
        self.model_name = model_name

    def _embed(self, text: str) -> List[float]:
        if not text.startswith("query:"):
            text = "query: " + text
        result = self.client.sentence_embedding(
            text,
            model=self.model_name
        )
        return result.embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

# Load prompt and context from file
with open("myinfo.txt", "r", encoding="utf-8") as f:
    content = f.read()

prompt_text = content.split("---PROMPT---")[1].split("---END---")[0].strip()
context_text = content.split("---CONTEXT---")[1].strip()

# Create prompt template
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_text
)

# Embed context using hosted embeddings
documents = [Document(page_content=context_text)]
embedding_model = HFHostedEmbeddings()
vector_db = FAISS.from_documents(documents, embedding_model)
retriever = vector_db.as_retriever()

# Build RetrievalQA chain
llm = TogetherLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": rag_prompt}
)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = qa_chain.invoke(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    import sys

    # Check if running in terminal mode
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        print("🧠 Terminal Chatbot (type 'exit' to quit)")
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            try:
                answer = qa_chain.invoke(question)
                print("Bot:", answer)
            except Exception as e:
                print("⚠️ Error:", e)
    else:
        # Run Flask app
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
