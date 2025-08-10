import os
import base64
import json
from typing import List

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ========= ENV =========
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")  # e.g. https://api.runpod.ai/v2/<endpoint_id>
DEFAULT_CKPT = os.getenv("CKPT_NAME", "flux1-dev-fp8.safetensors")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
    raise RuntimeError("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT env vars on Render.")

# ========= TEMPLATE PROMPTS =========
TEMPLATES = {
    "bedroom": "Framed artwork hanging in a cozy bedroom with sunlight filtering through linen curtains, photorealistic interior, soft natural light, realistic shadows, DSLR photo.",
    "gallery_wall": "Framed print on a gallery wall with spot lighting and minimal decor, photorealistic, clean plaster wall, accurate shadows.",
    "modern_lounge": "Framed artwork in a modern minimalist lounge above a sofa, natural window light, neutral palette, photorealistic.",
    "rustic_study": "Framed artwork in a rustic study with wooden shelves and a warm desk lamp, cozy lighting, photorealistic.",
    "kitchen": "Framed botanical print in a bright modern kitchen with plants, daylight, photorealistic."
}
NEGATIVE_PROMPT = "blurry, low detail, distorted, bad framing, artifacts, low quality, overexposed, underexposed"

# ========= FASTAPI =========
app = FastAPI(title="Mockup Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class BatchResponse(BaseModel):
    template: str
    prompt: str
    images: List[str]  # URLs or data URLs (normalized)

# ========= HTTP client with retries =========
def _build_session():
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

def _headers():
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

# ========= Helpers =========
def strip_data_url(b64_str: str) -> str:
    if ";base64," in b64_str:
        return b64_str.split(";base64,", 1)[1]
    return b64_str

def guess_mime(upload: UploadFile) -> str:
    # Prefer provided content_type, fallback to filename
    ct = (upload.content_type or "").lower()
    if ct in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        return "image/jpeg" if ct == "image/jpg" else ct
    name = (upload.filename or "").lower()
    if name.endswith(".png"): return "image/png"
    if name.endswith(".jpg") or name.endswith(".jpeg"): return "image/jpeg"
    if name.endswith(".webp"): return "image/webp"
    # default safe bet
    return "image/png"

def call_runsync(payload: dict, timeout_sec: int = 420) -> dict:
    """One-shot RunPod call (no polling)."""
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
    Normalize a variety of RunPod/ComfyUI output shapes into a list of displayable image strings.
    Preference order: URLs -> data URLs -> file paths (last resort).
    """
    out = (status_payload or {}).get("output") or {}

    results: List[str] = []

    # 1) {"output":{"images":[...]}}
    imgs = out.get("images")
    if isinstance(imgs, list):
        for it in imgs:
            if isinstance(it, dict):
                if it.get("url"):
                    results.append(it["url"])
                elif it.get("base64"):
                    results.append("data:image/png;base64," + it["base64"])
                elif it.get("content"):  # some wrappers use 'content' for base64
                    results.append("data:image/png;base64," + it["content"])
                elif it.get("path"):
                    results.append(it["path"])
            elif isinstance(it, str):
                if it.startswith("http"):
                    results.append(it)
                else:
                    results.append("data:image/png;base64," + it)
        if results:
            return results

    # 2) {"output":{"urls":[...]}}
    urls = out.get("urls")
    if isinstance(urls, list) and urls:
        return urls

    # 3) {"output":{"image_url":"..."}}
    if isinstance(out.get("image_url"), str) and out["image_url"]:
        return [out["image_url"]]

    # 4) {"output":{"base64":[...]}}
    b64s = out.get("base64")
    if isinstance(b64s, list) and b64s:
        return ["data:image/png;base64," + b for b in b64s if isinstance(b, str) and b]

    # 5) {"output":{"data":[{"images":[...]}, ...]}}
    data_arr = out.get("data")
    if isinstance(data_arr, list):
        for item in data_arr:
            if isinstance(item, dict) and isinstance(item.get("images"), list):
                for it in item["images"]:
                    if isinstance(it, dict):
                        if it.get("url"):
                            results.append(it["url"])
                        elif it.get("base64"):
                            results.append("data:image/png;base64," + it["base64"])
                        elif it.get("content"):
                            results.append("data:image/png;base64," + it["content"])
                    elif isinstance(it, str):
                        if it.startswith("http"):
                            results.append(it)
                        else:
                            results.append("data:image/png;base64," + it)
        if results:
            return results

    # 6) {"output":{"image_path":"..."}}
    if isinstance(out.get("image_path"), str) and out["image_path"]:
        return [out["image_path"]]

    # 7) Nothing recognized
    return []

# ========= ComfyUI WORKFLOW =========
def build_comfyui_workflow(prompt: str, art_image_data_url: str, seed: int) -> dict:
    """
    Stock ComfyUI nodes. Background is generated; your art is preserved (no diffusion).
    Placement: 1024x1024 canvas, 512x512 art centered (x=y=32 latent units = 256 px).
    """
    workflow_graph = {
        "100": {  # checkpoint loader
            "class_type": "CheckpointLoaderSimple",
            "inputs": { "ckpt_name": DEFAULT_CKPT }
        },
        "0": {  # LoadImage from uploaded set
            "class_type": "LoadImage",
            "inputs": { "image": "art.png", "upload": True }
        },
"1": {  # Resize to framed size
    "class_type": "ImageScale",
    "inputs": {
        "image": ["0", 0],
        "width": 512,
        "height": 512,
        "upscale_method": "lanczos",
        "keep_proportions": True,
        "crop": "disabled"   # <-- was False; must be "disabled" or "center"
    }
},
        "2": {  # VAEEncode artwork
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["100", 2]
            }
        },
        "3": {  # Positive prompt
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["100", 1],
                "text": prompt
            }
        },
        "4": {  # Negative prompt
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["100", 1],
                "text": NEGATIVE_PROMPT
            }
        },
        "5": {  # Empty background latent 1024x1024
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        },
        "6": {  # Generate background only
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
        "7": {  # Composite art onto background (no feather)
            "class_type": "LatentComposite",
            "inputs": {
                "samples_to": ["6", 0],   # background latent
                "samples_from": ["2", 0], # art latent
                "x": 32,                  # 256px / 8 px per latent
                "y": 32,
                "feather": 0,
                "tiled": False
            }
        },
        "8": {  # Decode to image
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["100", 2]
            }
        },
        "9": {  # Save output
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "mockup_out"
            }
        }
    }

    # IMPORTANT: your worker requires {name, image} and wants a data URL in 'image'
    return {
        "workflow": workflow_graph,
        "images": [
            { "name": "art.png", "image": art_image_data_url }
        ]
    }

# ========= ROUTES =========
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

    # read upload -> base64 (no prefix), then build proper data URL with correct MIME
    raw = await file.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    b64 = strip_data_url(b64)

    mime = guess_mime(file)  # e.g. image/png or image/jpeg
    data_url = f"data:{mime};base64,{b64}"

    prompt_text = TEMPLATES[template]

    images_all: List[str] = []
    # 5 variations via seed offsets
    for i in range(5):
        wf_input = build_comfyui_workflow(prompt_text, data_url, seed=1234567 + i)
        payload = { "input": { "return_type": "base64", **wf_input } }
        result = call_runsync(payload, timeout_sec=420)

        # TEMP: log first result for debugging
        if i == 0:
            try:
                print("RUNPOD_RAW_SAMPLE:", json.dumps(result)[:4000])
            except Exception:
                print("RUNPOD_RAW_SAMPLE: <non-serializable>")

        outs = extract_images_from_output(result)
        images_all.append(outs[0] if outs else "MISSING")

    return BatchResponse(template=template, prompt=prompt_text, images=images_all)
