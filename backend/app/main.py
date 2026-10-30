
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

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
async def chat(request: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI API key not configured"
        )

    try:
        client = AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are THESIS AI, an investment "
                        "research assistant. Respond in "
                        "Vietnamese. Explain financial "
                        "analysis, risks, and uncertainty. "
                        "Never guarantee investment returns."
                    )
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ]
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI provider request failed"
        )
