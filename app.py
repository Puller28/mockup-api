# app.py
import io
import os
import json
import time
import base64
import secrets
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI(
    title="Mockup API",
    description="Upload an image and generate wall-art mockups via RunPod (mask worker + ComfyUI).",
    version="2.4"
)

# ---------------- Env ----------------
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT    = os.getenv("RUNPOD_MASK_ENDPOINT")      # .../<MASK_ID>/runsync
COMFY_ENDPOINT   = os.getenv("RUNPOD_COMFY_ENDPOINT")     # .../<COMFY_EXEC_ID>/run  (async)
COMFY_MODEL      = os.getenv("RUNPOD_COMFY_MODEL", "v1-5-pruned-emaonly.safetensors")

def _assert_env():
    missing = []
    if not RUNPOD_API_KEY: missing.append("RUNPOD_API_KEY")
    if not MASK_ENDPOINT:  missing.append("RUNPOD_MASK_ENDPOINT")
    if not COMFY_ENDPOINT: missing.append("RUNPOD_COMFY_ENDPOINT")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env: {', '.join(missing)}")

# ------------- Style presets ---------------
STYLE_PRESETS: Dict[str, str] = {
    "minimal_gallery":    "minimalist art gallery wall, white walls, professional gallery lighting, subtle wall texture, premium interior photography, neutral palette",
    "modern_living_room": "modern Scandinavian living room, white wall with subtle texture, light oak floor, soft natural daylight from large windows, realistic photography, clean composition",
    "reading_nook":       "cozy reading nook, framed art above armchair, warm ambient lamp light plus soft daylight, realistic shadows, inviting neutral tones",
    "industrial_loft":    "industrial loft, exposed brick feature wall, concrete floor, tall metal windows with daylight, cinematic yet realistic shadows",
    "scandi_bedroom":     "Scandinavian bedroom, framed art above bed, pastel wall, linen bedding, bright diffused daylight, airy and realistic",
    "elegant_hallway":    "elegant hallway, cream wall with subtle texture, bright natural side light, professional interior photo, balanced tones"
}
DEFAULT_STYLE_KEY = "minimal_gallery"
DEFAULT_PROMPT    = STYLE_PRESETS[DEFAULT_STYLE_KEY]

def build_prompt(style_value: str | None) -> str:
    if not style_value:
        return DEFAULT_PROMPT
    s = style_value.strip()
    return STYLE_PRESETS.get(s, s)  # preset key or free text

# ------------- Image helpers ---------------
def clamp_image(img: Image.Image, max_side=1280) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return img.resize((int(w * s), int(h * s)), Image.LANCZOS)

def b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def b64_jpeg(img: Image.Image, q=90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def normalize_mask_b64(mask_b64: str) -> str:
    """Decode -> sanitize -> re-encode mask as a clean RGBA PNG with alpha channel."""
    try:
        raw = base64.b64decode(mask_b64)
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        if "A" in im.getbands():
            alpha = im.getchannel("A")
        else:
            alpha = im.convert("L")
        clean = Image.new("RGBA", im.size, (255, 255, 255, 0))
        clean.putalpha(alpha)
        buf = io.BytesIO()
        clean.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return mask_b64

# ------------- HTTP helpers ---------------
def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def _post_sync(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(endpoint, headers=_headers(), json=payload, timeout=240)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
    return r.json()

def _post_async(endpoint_run: str, payload: Dict[str, Any], timeout_s: int = 240, poll_every: float = 1.5) -> Dict[str, Any]:
    r = requests.post(endpoint_run, headers=_headers(), json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint_run}: {r.text}")
    job = r.json()
    job_id = job.get("id") or job.get("output", {}).get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod /run did not return a job id: {job}")
    base = endpoint_run.rsplit("/", 1)[0]
    status_url = f"{base}/status/{job_id}"
    started = time.time()
    while True:
        s = requests.get(status_url, headers=_headers(), timeout=60)
        if s.status_code != 200:
            raise HTTPException(status_code=s.status_code, detail=f"RunPod status error {status_url}: {s.text}")
        js = s.json()
        st = (js.get("status") or "").upper()
        if st in ("COMPLETED", "COMPLETEDWITHERROR"):
            return js
        if st in ("FAILED", "CANCELLED"):
            raise HTTPException(status_code=502, detail=f"RunPod job failed {job_id}: {js}")
        if time.time() - started > timeout_s:
            raise HTTPException(status_code=504, detail=f"RunPod job timed out {job_id}")
        time.sleep(poll_every)

def call_runpod(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if endpoint.rstrip("/").endswith("/runsync"):
        return _post_sync(endpoint, payload)
    if endpoint.rstrip("/").endswith("/run"):
        return _post_async(endpoint, payload)
    return _post_sync(endpoint, payload)

# ---------- Mask worker ----------
def call_mask_worker(img_b64: str, mode: str = "preview") -> str:
    payload = {"input": {"image_b64": img_b64, "mode": mode}}
    data = call_runpod(MASK_ENDPOINT, payload)
    out = data.get("output") or {}
    mask_b64 = out.get("mask_b64") or data.get("mask_b64")
    if not mask_b64:
        raise HTTPException(status_code=502, detail=f"Mask worker returned no mask: {data}")
    return mask_b64

# ---------- ComfyUI workflow ----------
def comfy_workflow(seed: int, prompt: str) -> Dict[str, Any]:
    return {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": COMFY_MODEL}
        },
        "img":   {"class_type": "LoadImage",     "inputs": {"image": "__b64_img__"}},
        "mask":  {"class_type": "LoadImageMask", "inputs": {"image": "__b64_mask__", "channel": "alpha"}},
        "pos":   {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
        "neg":   {"class_type": "CLIPTextEncode", "inputs": {"text": "",      "clip": ["ckpt", 1]}},
        "enc": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["img", 0], "mask": ["mask", 0], "vae": ["ckpt", 2]}
        },
        "noise": {"class_type": "RandomNoise", "inputs": {"seed": seed}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["enc", 0],
                "noise": ["noise", 0],
                "mask": ["enc", 1],
                "steps": 22,
                "cfg": 7.5,
                "sampler_name": "euler"
            }
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]}},
        "out":    {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}}
    }

def call_comfy(img_b64_for_comfy: str, mask_b64: str, prompt: str, seed: int) -> List[str]:
    wf = json.loads(
        json.dumps(comfy_workflow(seed, prompt))
        .replace("__b64_img__", img_b64_for_comfy)
        .replace("__b64_mask__", mask_b64)
    )
    payload = {"input": {"return_type": "base64", "workflow": wf}}
    data = call_runpod(COMFY_ENDPOINT, payload)
    out = data.get("output") or {}
    images = out.get("images", [])
    if not images and isinstance(out, list):
        images = [{"image": x} for x in out]
    if not images:
        raise HTTPException(status_code=502, detail=f"Comfy returned no images: {data}")
    return [e["image"] for e in images if isinstance(e, dict) and "image" in e]

# -------------- Routes --------------
@app.get("/debug/env")
def debug_env():
    return {
        "has_api_key": bool(RUNPOD_API_KEY),
        "mask_endpoint": MASK_ENDPOINT,
        "comfy_endpoint": COMFY_ENDPOINT,
        "comfy_model": COMFY_MODEL
    }

@app.get("/styles")
def styles():
    return {"default": DEFAULT_STYLE_KEY, "presets": STYLE_PRESETS}

@app.post("/mask/test")
async def mask_test(file: UploadFile = File(...), mode: str = Form("preview")):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    mask_b64 = normalize_mask_b64(mask_b64)
    return {"mode": mode, "mask_b64": mask_b64}

@app.post("/preview/json")
async def preview_json(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY),
    mode: str = Form("preview")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)

    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    mask_b64 = normalize_mask_b64(mask_b64)

    img_small = clamp_image(img, max_side=1280)
    img_b64_for_comfy = b64_jpeg(img_small, q=90)

    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for s in seeds:
        imgs = call_comfy(img_b64_for_comfy, mask_b64, prompt, seed=s)
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})

    return {"mode": mode, "style": style, "prompt_used": prompt, "count": len(results), "results": results}

@app.post("/preview/html", response_class=HTMLResponse)
async def preview_html(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY),
    mode: str = Form("preview")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)

    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    mask_b64 = normalize_mask_b64(mask_b64)

    img_small = clamp_image(img, max_side=1280)
    img_b64_for_comfy = b64_jpeg(img_small, q=90)

    seeds = [secrets.randbits(32) for _ in range(5)]
    imgs64: List[str] = []
    for s in seeds:
        out = call_comfy(img_b64_for_comfy, mask_b64, prompt, seed=s)
        if out:
            imgs64.append(out[0])

    html = "<h1>Generated Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for img64 in imgs64:
        html += f"<figure style='margin:8px'><img src='data:image/png;base64,{img64}' style='width:300px;margin:5px;display:block'/></figure>"
    html += "</div>"
    return HTMLResponse(content=html)
