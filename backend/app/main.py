from fastapi import FastAPI

app = FastAPI(title="THESIS AI")

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
