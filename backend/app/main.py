
import os
import logging

logger = logging.getLogger("uvicorn.error")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="THESIS AI")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {
        "name": "THESIS AI",
        "status": "online",
        "message": "THESIS AI backend is ready"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key not configured"
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are THESIS AI, a Vietnamese "
                    "investment research assistant. "
                    "Explain financial concepts clearly, "
                    "analyze risks and uncertainty, and "
                    "never guarantee investment returns."
                )
            )
        )

        return {
            "reply": response.text
        }
    
    except Exception as e:
        logger.error(
            "Gemini error: code=%s, status=%s",
            getattr(e, "code", "unknown"),
            getattr(e, "status", "unknown")
        )
        raise HTTPException(
            status_code=502,
            detail="Gemini API request failed"
        )

@app.get("/debug-models")
def debug_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(503, "API key missing")

    try:
        client = genai.Client(api_key=api_key)
        return {
            "models": [
                m.name
                for m in client.models.list()
                if "generateContent" in (
                    m.supported_actions or []
                )
            ]
        }
    except Exception as e:
        logger.error(
            "Model list error: %s, code=%s",
            type(e).__name__,
            getattr(e, "code", "unknown")
        )
        raise HTTPException(502, "Model listing failed")
