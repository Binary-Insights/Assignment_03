from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import requests

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat: ChatRequest):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": chat.message}]
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/openai-health")
def openai_health():
    try:
        response = requests.get("https://status.openai.com/api/v2/status.json", timeout=5)
        data = response.json()
        status = data.get("status", {}).get("description", "Unknown")
        return {"openai_status": status}
    except Exception as e:
        return {"error": str(e)}
