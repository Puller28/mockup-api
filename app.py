import io
import base64
import os
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI(
    title="Mockup API",
    description="Upload an image and generate mockups via RunPod",
    version="1.0"
)

# Environment variables
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT = os.getenv("RUNPOD_MASK_ENDPOINT")  # e.g., https://api.runpod.ai/v2/<MASK_ID>/runsync
COMFY_ENDPOINT = os.getenv("RUNPOD_COMFY_ENDPOINT")  # e.g., https://api.runpod.ai/v2/<COMFY_ID>/runsync


# Helpers
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_runpod(endpoint: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
    return r.json()


def call_mask_worker(img_b64: str, mode: str = "preview") -> str:
    payload = {"input": {"image_b64": img_b64, "mode": mode}}
    data = call_runpod(MASK_ENDPOINT, payload)
    return data.get("output", {}).get("mask_b64")


def call_comfy(img_b64: str, mask_b64: str, style: str) -> list:
    workflow = {
        "img": {"class_type": "LoadImage", "inputs": {"image": img_b64}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": mask_b64, "channel": "alpha"}},
        "gen": {"class_type": "KSampler", "inputs": {
            "model": ["checkpoint", 0],
            "positive": [style, 0],
            "negative": ["", 0],
            "latent_image": ["img", 0],
            "mask": ["mask", 0],
            "steps": 20,
            "cfg": 7.5,
            "sampler_name": "euler"
        }},
        "out": {"class_type": "SaveImage", "inputs": {"images": ["gen", 0]}}
    }
    payload = {"input": {"return_type": "base64", "workflow": workflow}}
    data = call_runpod(COMFY_ENDPOINT, payload)
    return [img["image"] for img in data.get("output", {}).get("images", [])]


# API routes
@app.post("/preview/json", summary="Preview mockups (JSON)")
async def preview_json(file: UploadFile = File(...), style: str = Form("modern living room")):
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    img_b64 = pil_to_b64(img)
    mask_b64 = call_mask_worker(img_b64)
    images = call_comfy(img_b64, mask_b64, style)
    return {"style": style, "count": len(images), "images": images}


@app.post("/preview/html", summary="Preview mockups (HTML)")
async def preview_html(file: UploadFile = File(...), style: str = Form("modern living room")):
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    img_b64 = pil_to_b64(img)
    mask_b64 = call_mask_worker(img_b64)
    images = call_comfy(img_b64, mask_b64, style)
    html = "<h1>Generated Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for img64 in images:
        html += f"<img src='data:image/png;base64,{img64}' style='width:300px;margin:5px'/>"
    html += "</div>"
    return HTMLResponse(content=html)


@app.post("/mask/test", summary="Test mask worker (returns mask only)")
async def mask_test(file: UploadFile = File(...), mode: str = Form("preview")):
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    img_b64 = pil_to_b64(img)
    mask_b64 = call_mask_worker(img_b64, mode)
    return {"mask_b64": mask_b64}


@app.get("/debug/env", summary="Debug environment variables")
def debug_env():
    return {
        "has_api_key": bool(RUNPOD_API_KEY),
        "mask_endpoint": MASK_ENDPOINT,
        "comfy_endpoint": COMFY_ENDPOINT
    }
