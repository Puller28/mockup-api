# app.py
import os
import json
import base64
from typing import List, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------- ENV ----------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")  # e.g. https://api.runpod.ai/v2/<endpoint_id>
DEFAULT_CKPT = os.getenv("CKPT_NAME", "flux1-dev-fp8.safetensors")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
    raise RuntimeError("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT env vars on Render.")

# ---------- PROMPTS ----------
TEMPLATES: Dict[str, str] = {
    "bedroom": "Framed artwork hanging in a cozy bedroom with sunlight filtering through linen curtains, photorealistic interior, soft natural light, realistic shadows, DSLR photo.",
    "gallery_wall": "Framed print on a gallery wall with spot lighting and minimal decor, photorealistic, clean plaster wall, accurate shadows.",
    "modern_lounge": "Framed artwork in a modern minimalist lounge above a sofa, natural window light, neutral palette, photorealistic.",
    "rustic_study": "Framed artwork in a rustic study with wooden shelves and a warm desk lamp, cozy lighting, photorealistic.",
    "kitchen": "Framed botanical print in a bright modern kitchen with plants, daylight, photorealistic.",
}
NEGATIVE_PROMPT = "blurry, low detail, distorted, bad framing, artifacts, low quality, overexposed, underexposed"

# ---------- APP ----------
app = FastAPI(title="Mockup Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class BatchResponse(BaseModel):
    template: str
    prompt: str
    images: List[str]  # data URLs or http URLs

# ---------- HTTP SESSION ----------
def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

SESSION = _build_session()

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

# ---------- HELPERS ----------
def strip_data_url(b64_str: str) -> str:
    if ";base64," in b64_str:
        return b64_str.split(";base64,", 1)[1]
    return b64_str

def to_data_url(b64_png: str) -> str:
    # Guard: if it already looks like a URL, return as-is
    if b64_png.startswith("http://") or b64_png.startswith("https://"):
        return b64_png
    if b64_png.startswith("data:image/"):
        return b64_png
    return "data:image/png;base64," + b64_png

def call_runsync(payload: dict, timeout_sec: int = 420) -> dict:
    url = f"{RUNPOD_ENDPOINT}/runsync"
    try:
        r = SESSION.post(url, json=payload, headers=_headers(), timeout=timeout_sec)
    except (ChunkedEncodingError, ProtocolError, ConnectionError, ReadTimeout) as e:
        raise HTTPException(status_code=502, detail=f"RunPod connect error: {e}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail={"runpod_runsync_error": r.text})
    try:
        return r.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail={"runpod_runsync_error": "Invalid JSON from RunPod", "raw": r.text})

def extract_images_from_output(status_payload: dict) -> List[str]:
    """
    Normalize the many shapes RunPod/ComfyUI workers return into a list of *data URLs* (or http URLs).
    """
    out = (status_payload or {}).get("output") or {}
    results: List[str] = []

    imgs = out.get("images")
    if isinstance(imgs, list):
        for it in imgs:
            if isinstance(it, dict):
                if it.get("url"):
                    results.append(it["url"])
                elif it.get("base64"):
                    results.append(to_data_url(it["base64"]))
                elif it.get("content"):   # some workers use "content"
                    results.append(to_data_url(it["content"]))
                elif it.get("data"):      # some workers use "data"
                    results.append(to_data_url(it["data"]))
                elif it.get("image"):     # rare key name
                    results.append(to_data_url(it["image"]))
            elif isinstance(it, str):
                if it.startswith("http"):
                    results.append(it)
                else:
                    results.append(to_data_url(it))
        if results:
            return results

    urls = out.get("urls")
    if isinstance(urls, list) and urls:
        return urls

    if isinstance(out.get("image_url"), str) and out["image_url"]:
        return [out["image_url"]]

    b64s = out.get("base64")
    if isinstance(b64s, list) and b64s:
        return [to_data_url(b) for b in b64s if isinstance(b, str) and b]

    data_arr = out.get("data")
    if isinstance(data_arr, list):
        for item in data_arr:
            if isinstance(item, dict) and isinstance(item.get("images"), list):
                for it in item["images"]:
                    if isinstance(it, dict):
                        if it.get("url"):
                            results.append(it["url"])
                        elif it.get("base64"):
                            results.append(to_data_url(it["base64"]))
                        elif it.get("content"):
                            results.append(to_data_url(it["content"]))
                    elif isinstance(it, str):
                        if it.startswith("http"):
                            results.append(it)
                        else:
                            results.append(to_data_url(it))
        if results:
            return results

    # fallback nothing recognized
    return results

# ---------- WORKFLOW ----------
def build_comfyui_workflow(prompt: str, art_b64_no_prefix: str, seed: int) -> dict:
    """
    Background via diffusion; artwork preserved and composited via LatentComposite.
    1024x1024 canvas, artwork scaled to 512x512 (keep aspect) and centered.
    """
    return {
        "workflow": {
            "100": {  # checkpoint
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": DEFAULT_CKPT}
            },
            "0": {  # uploaded art
                "class_type": "LoadImage",
                "inputs": {"image": "art.png", "upload": True}
            },
            "1": {  # scale artwork (keep aspect), crop disabled
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["0", 0],
                    "width": 512,
                    "height": 512,
                    "upscale_method": "lanczos",
                    "crop": "disabled",          # <-- important for your Comfy version
                    "keep_proportions": True
                }
            },
            "2": {  # encode art to latent
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["1", 0], "vae": ["100", 2]}
            },
            "3": {  # positive prompt
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["100", 1], "text": prompt}
            },
            "4": {  # negative prompt
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["100", 1], "text": NEGATIVE_PROMPT}
            },
            "5": {  # background latent 1024
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
            },
            "6": {  # generate background only
                "class_type": "KSampler",
                "inputs": {
                    "model": ["100", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": seed,
                    "steps": 22,
                    "cfg": 6.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "7": {  # composite art onto background (centered)
                "class_type": "LatentComposite",
                "inputs": {
                    "samples_to": ["6", 0],
                    "samples_from": ["2", 0],
                    "x": 32,   # 256 px / 8
                    "y": 32,
                    "feather": 0,
                    "tiled": False
                }
            },
            "8": {  # decode to image
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["100", 2]}
            },
            "9": {  # save (some workers also echo this as output.images)
                "class_type": "SaveImage",
                "inputs": {"images": ["8", 0], "filename_prefix": "mockup_out"}
            }
        },
        # NEW: use "name"/"image" shape many RunPod workers expect
        "images": [
            {"name": "art.png", "image": art_b64_no_prefix}
        ]
    }

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"message": "Mockup API running"}

@app.get("/debug/env")
def debug_env():
    return {
        "RUNPOD_ENDPOINT": RUNPOD_ENDPOINT,
        "has_api_key": bool(RUNPOD_API_KEY),
        "routes": [r.path for r in app.routes],
    }

@app.post("/batch", response_model=BatchResponse)
async def batch(template: str = Form(...), file: UploadFile = File(...)):
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Invalid template. Available: {list(TEMPLATES.keys())}")

    raw = await file.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    b64 = strip_data_url(b64)

    prompt_text = TEMPLATES[template]
    images_all: List[str] = []

    for i in range(5):
        wf_input = build_comfyui_workflow(prompt_text, b64, seed=1234567 + i)
        payload = {"input": {"return_type": "base64", **wf_input}}
        result = call_runsync(payload, timeout_sec=420)

        if i == 0:
            try:
                print("RUNPOD_RAW_SAMPLE:", json.dumps(result)[:4000])
            except Exception:
                print("RUNPOD_RAW_SAMPLE: <non-serializable>")

        outs = extract_images_from_output(result)
        images_all.append(outs[0] if outs else "MISSING")

    # 👉 images_all now contains data URLs (or http URLs) ready to view/copy from Swagger
    return BatchResponse(template=template, prompt=prompt_text, images=images_all)

@app.post("/batch/html", response_class=HTMLResponse)
async def batch_html(template: str = Form(...), file: UploadFile = File(...)):
    """Optional: returns a tiny HTML gallery with <img> tags for instant visual check."""
    resp = await batch(template, file)  # reuse core logic
    html_imgs = "".join(f'<div style="margin:8px 0"><img style="max-width:600px" src="{u}"><br><small>{u[:80]}...</small></div>' for u in resp.images)
    return f"<h3>{resp.template}</h3><p>{resp.prompt}</p>{html_imgs}"
