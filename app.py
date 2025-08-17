from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from langchain_core.chains import LLMChain
from openai import OpenAI
from typing import Optional, List
import os
import dotenv
import logging

# Load environment variables
dotenv.load_dotenv()

# Get Hugging Face API key securely
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("❌ HF_API_KEY not found in environment variables.")

logging.basicConfig(level=logging.INFO)

# Initialize Together.ai client
together_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY
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

# Build LLMChain
llm = TogetherLLM()
qa_chain = LLMChain(
    llm=llm,
    prompt=rag_prompt
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
    try:
        answer = qa_chain.invoke({"context": context_text, "question": question})
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error("❌ Error during QA invocation", exc_info=True)
        return jsonify({"error": "Failed to generate answer"}), 500

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
                answer = qa_chain.invoke({"context": context_text, "question": question})
                print("Bot:", answer)
            except Exception as e:
                print("⚠️ Error:", e)
    else:
        # Run Flask app
        port = int(os.environ.get("PORT", 5000))

        app.run(host="0.0.0.0", port=port)
