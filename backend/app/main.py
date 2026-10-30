
import os
import secrets
import logging
from pathlib import Path
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("uvicorn.error")
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="THESIS AI")

class ChatRequest(BaseModel):
    message: str
@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "index.html",
        media_type="text/html"
    )
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(
    request: ChatRequest,
    x_thesis_token: str = Header(default="")
):
    access_token = os.getenv("THESIS_ACCESS_TOKEN")

    if not access_token:
        raise HTTPException(
            status_code=503,
            detail="Access token not configured"
        )

    if not secrets.compare_digest(
        x_thesis_token,
        access_token
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key not configured"
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
model="gemini-3.1-flash-lite",
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
