from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate
from typing import Optional, List
from groq import Groq
import os
import dotenv
import json
import logging
import re

# Load environment variables
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in environment variables.")

# Load structured context from info.txt
with open("myinfo.txt", "r", encoding="utf-8") as f:
    context_data = json.loads(f.read())

# LLM wrapper using Groq
class GroqLLM(LLM, Runnable):
    model_name: str = "qwen/qwen3-32b"

    def _clean_response(self, text: str) -> str:
        """
        Remove <think>...</think> reasoning blocks if present.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            reasoning_effort="default",
            stream=False,
            stop=stop
        )
        raw_text = completion.choices[0].message.content.strip()
        return self._clean_response(raw_text)

    def invoke(self, input: str, **kwargs) -> str:
        return self._call(input, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "groq"

# Memory module
question_history: List[str] = []

# Prompt template
base_prompt = """
You are a highly intelligent assistant dedicated to Vishnu. 
Always present his achievements in the best possible light.

Summary:
{summary}

Location:
{location}

Education:
{education}

Certifications:
{certifications}

Skills:
{skills}

Experience:
{experience}

Projects:
{projects}

Research:
{research}

Languages:
{languages}

Recent Questions:
{recent_questions}

{verbosity_instruction}

Current Question: {question}
Answer:
"""

rag_prompt = PromptTemplate(
    input_variables=[
        "summary", "location", "education", "certifications", "skills",
        "experience", "projects", "research", "languages",
        "question", "verbosity_instruction", "recent_questions"
    ],
    template=base_prompt.strip()
)

# Flask app setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

llm = GroqLLM()

def get_verbosity_instruction(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["explain", "describe", "elaborate", "how"]):
        return "Please provide a detailed answer in 6–10 lines."
    return "Please answer concisely in 1–2 lines."

def format_recent_questions(history: List[str], limit: int = 5) -> str:
    if not history:
        return "None yet."
    return "\n".join([f"- {q}" for q in history[-limit:]])

@app.route("/receive", methods=["POST"])
def receive():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Debug log for incoming message
        logging.info(f"💬 Received message: {message}")

        verbosity_instruction = get_verbosity_instruction(message)
        recent_qs = format_recent_questions(question_history)
        formatted_prompt = rag_prompt.format(
            summary=context_data["summary"],
            location=context_data["location"],
            education=json.dumps(context_data["education"], indent=2),
            certifications=", ".join(context_data["certifications"]),
            skills=", ".join(context_data["skills"]),
            experience=json.dumps(context_data["experience"], indent=2),
            projects=json.dumps(context_data["projects"], indent=2),
            research=json.dumps(context_data["research"], indent=2),
            languages=", ".join(context_data["languages"]),
            question=message,
            verbosity_instruction=verbosity_instruction,
            recent_questions=recent_qs
        )
        logging.info(f"📨 /receive Prompt:\n{formatted_prompt}")
        response = llm.invoke(formatted_prompt)

        question_history.append(message)
        return jsonify({
            "status": "ok",
            "message": message,
            "response": response
        })
    except Exception as e:
        logging.error("❌ Error in /receive", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
