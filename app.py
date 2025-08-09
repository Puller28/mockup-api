import os
import base64
import time
import requests
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Load environment variables
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")  # e.g. https://xxxx.api.runpod.ai

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
    raise RuntimeError("RUNPOD_API_KEY and RUNPOD_ENDPOINT must be set as environment variables")

# Templates: prompt text for each room style
TEMPLATES = {
    "bedroom": "A cozy, photorealistic bedroom with soft window light, neutral tones, and a framed artwork hanging above the bed. Realistic textures, natural shadows, DSLR photography style.",
    "gallery_wall": "A modern art gallery wall with minimal decor, soft spot lighting, and a framed artwork. Photorealistic interior design photography."
}

# FastAPI init
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MockupResponse(BaseModel):
    template: str
    prompt: str
    images: List[str]

@app.post("/batch", response_model=MockupResponse)
async def generate_batch(template: str = Form(...), file: UploadFile = None):
    """
    Generates 5 mockups for the given template and uploaded image.
    """
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Invalid template. Available: {list(TEMPLATES.keys())}")

    if not file:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    # Read and encode uploaded image
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt_text = TEMPLATES[template]

    # Build the RunPod payload
    payload = {
        "input": {
            "workflow": build_workflow(prompt_text, img_b64, num_outputs=5)
        }
    }

    # Submit job to RunPod
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    submit_resp = requests.post(f"{RUNPOD_ENDPOINT}/run", json=payload, headers=headers)
    if submit_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"RunPod job submission failed: {submit_resp.text}")

    job_id = submit_resp.json().get("id")
    if not job_id:
        raise HTTPException(status_code=500, detail="No job ID returned from RunPod")

    # Poll until complete
    status_url = f"{RUNPOD_ENDPOINT}/status/{job_id}"
    while True:
        status_resp = requests.get(status_url, headers=headers)
        if status_resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"RunPod status check failed: {status_resp.text}")

        status_data = status_resp.json()
        status = status_data.get("status")
        if status == "COMPLETED":
            output_urls = extract_urls(status_data)
            return {
                "template": template,
                "prompt": prompt_text,
                "images": output_urls
            }
        elif status == "FAILED":
            raise HTTPException(status_code=500, detail="RunPod job failed")

        time.sleep(3)  # wait before polling again

def build_workflow(prompt: str, img_b64: str, num_outputs: int):
    """
    Returns the workflow JSON with uploaded image composited into generated background.
    - Uses hybrid composite method, no transformation of artwork.
    - Generates multiple outputs in one job.
    """
    return {
        "nodes": [
            {"id": 0, "type": "LoadImage", "inputs": {"image": img_b64}},
            {"id": 1, "type": "ImageScale", "inputs": {"width": 512, "height": 512, "keep_aspect": True, "image": "#0"}},
            {"id": 2, "type": "PromptToLatent", "inputs": {"prompt": prompt}},
            {"id": 3, "type": "LatentComposite", "inputs": {"background": "#2", "foreground": "#1", "x": 400, "y": 300}},
            {"id": 4, "type": "LatentToImage", "inputs": {"latent": "#3", "num_outputs": num_outputs}}
        ]
    }

def extract_urls(status_data):
    """
    Extracts image URLs from RunPod job result.
    """
    outputs = status_data.get("output", {}).get("images", [])
    return [img.get("url") for img in outputs if img.get("url")]
