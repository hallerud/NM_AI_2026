# main.py
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/health")
def health():
    """
    Health check endpoint for the competition validator.
    """
    return {"status": "ok"}

@app.post("/solve")
async def solve(request: dict):
    """
    Main endpoint for competition tasks.
    Receives prompt and credentials, and returns a status.
    """
    prompt = request.get("prompt", "")
    credentials = request.get("tripletex_credentials", {})

    # Your AI agent logic will go here:
    # 1. Parse the prompt
    # 2. Call external APIs (e.g., Tripletex) using provided credentials
    # 3. Implement your solution to the accounting task

    print(f"Received prompt: {prompt}")
    print(f"Received credentials (Tripletex): {credentials}")

    # For now, we'll just return a 'completed' status.
    # You'll replace this with your actual task completion logic.
    return {"status": "completed", "message": "Task received and processed by skeleton."}

if __name__ == "__main__":
    # Get port from environment variable, default to 8080 for Cloud Run
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

