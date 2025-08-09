from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import requests
import os
import uuid
import time

# ===== CONFIG =====
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")  # e.g., https://xxxxx.api.runpod.ai/v2/worker-id/run
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

TEMPLATES = {
    "bedroom": "Framed artwork hanging in a cozy bedroom with sunlight filtering through linen curtains",
    "gallery_wall": "Framed print on a gallery wall with spot lighting and minimal decor",
    "modern_lounge": "Framed abstract painting in a modern minimalist lounge with natural lighting",
    "rustic_study": "Framed vintage map in a rustic study with wooden shelves and warm desk lamp lighting",
    "kitchen": "Framed botanical print in a bright, modern kitchen with potted plants"
}

# ===== FASTAPI =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class BatchResponse(BaseModel):
    urls: List[str]

# ===== UTIL =====
def image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def poll_runpod(job_id: str, timeout: int = 120):
    """Poll RunPod until job completes or fails."""
    status_url = RUNPOD_ENDPOINT.replace("/run", f"/status/{job_id}")
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    start_time = time.time()

    while time.time() - start_time < timeout:
        r = requests.get(status_url, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"RunPod status error: {r.text}")
        data = r.json()
        if data.get("status") == "COMPLETED":
            return data
        elif data.get("status") == "FAILED":
            raise HTTPException(status_code=500, detail=f"RunPod job failed: {data}")
        time.sleep(3)

    raise HTTPException(status_code=500, detail="RunPod job polling timed out")

def runpod_request(prompt: str, base64_image: str):
    # ===== Hybrid composite workflow JSON =====
    workflow = {
        "nodes": [
            # This is where your LAST WORKING JSON goes, unchanged
            # Replace this with the exact JSON we tested that produces correct background + image placement
        ]
    }

    payload = {
        "input": {
            "workflow": workflow,
            "prompt": prompt,
            "image": base64_image
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }

    r = requests.post(RUNPOD_ENDPOINT, json=payload, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"RunPod error: {r.text}")

    job_id = r.json().get("id")
    if not job_id:
        raise HTTPException(status_code=500, detail="RunPod did not return a job ID")

    result = poll_runpod(job_id)
    return result

# ===== ROUTES =====
@app.post("/batch", response_model=BatchResponse)
async def generate_batch(template: str = Form(...), file: UploadFile = File(...)):
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail="Invalid template name")

    # Save uploaded file
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    base64_img = image_to_base64(tmp_path)
    prompt = TEMPLATES[template]

    urls = []
    for _ in range(5):  # 5 variations
        result = runpod_request(prompt, base64_img)
        try:
            img_url = result.get("output", {}).get("image_url", None)
            if img_url:
                urls.append(img_url)
            else:
                urls.append("MISSING")
        except Exception:
            urls.append("MISSING")

    return {"urls": urls}

@app.get("/")
def root():
    return {"message": "Mockup API running"}
