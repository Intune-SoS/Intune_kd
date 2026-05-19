"""
FastAPI application for LLM inference.
Workers run separately: eval_first.py and eval_finetune.py
"""
import os
import sys
import subprocess
import time
import requests
import logging
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gemma-finetuned"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500

app = FastAPI(title="LLM Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:5174","http://localhost:5175", "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    response: str
    model: str


def is_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_ollama_service() -> bool:
    if not is_ollama_running():
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            for _ in range(10):
                if is_ollama_running():
                    return True
                time.sleep(1)
            return False
        except:
            return False
    return True


def generate_response(prompt: str) -> str:
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": DEFAULT_TEMPERATURE, "num_predict": DEFAULT_MAX_TOKENS}
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")


@app.get("/")
async def root():
    return {"message": "LLM Inference API", "version": "1.0.0", "model": MODEL_NAME}

@app.options("/generate")
async def generate_options():
    return Response(status_code=200)


@app.post("/generate", response_model=PromptResponse)
async def generate(request: PromptRequest):
    try:
        if not start_ollama_service():
            raise HTTPException(status_code=503, detail="Failed to start Ollama service")
        
        response_text = generate_response(prompt=request.prompt)

        # Insert to Supabase
        try:
            supabase = get_supabase_client()
            payload = {
                "input": request.prompt,
                "actual_output": response_text,
                "status_eval_first": "created"
            }
            supabase.table("intune_db").insert(payload).execute()
            logger.info(f"Inserted record to intune_db")
        except Exception as db_e:
            logger.error(f"Failed to write to Supabase: {db_e}")

        return PromptResponse(response=response_text, model=MODEL_NAME)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
