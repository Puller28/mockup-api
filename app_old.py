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
    description="Upload an image and generate wall-art mockups via RunPod (mask worker + exec inpainting worker).",
    version="3.0"
)

# ---------------- Env ----------------
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT    = os.getenv("RUNPOD_MASK_ENDPOINT")      # .../<MASK_ID>/runsync
COMFY_ENDPOINT   = os.getenv("RUNPOD_COMFY_ENDPOINT")     # .../<EXEC_ID>/run  OR  .../<EXEC_ID>/runsync

def _assert_env():
    missing = []
    if not RUNPOD_API_KEY: missing.append("RUNPOD_API_KEY")
    if not MASK_ENDPOINT:  missing.append("RUNPOD_MASK_ENDPOINT")
    if not COMFY_ENDPOINT: missing.append("RUNPOD_COMFY_ENDPOINT")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env: {', '.join(missing)}")

# ------------- Style presets ---------------
STYLE_PRESETS: Dict[str, str] = {
    "minimal_gallery":    "minimalist gallery wall, white walls, professional gallery lighting, subtle wall texture, premium interior photography, neutral palette",
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
    """Keep request small for serverless endpoints."""
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

# ------------- HTTP helpers ---------------
def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def _post_sync(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(endpoint, headers=_headers(), json=payload, timeout=240)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
    return r.json()

def _post_async(endpoint_run: str, payload: Dict[str, Any], timeout_s: int = 240, poll_every: float = 1.5) -> Dict[str, Any]:
    # submit job
    try:
        r = requests.post(endpoint_run, headers=_headers(), json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint_run}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint_run}: {r.text}")

    job = r.json()
    job_id = job.get("id") or job.get("output", {}).get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod /run did not return a job id: {job}")

    base = endpoint_run.rsplit("/", 1)[0]  # strip /run
    status_url = f"{base}/status/{job_id}"

    started = time.time()
    while True:
        try:
            s = requests.get(status_url, headers=_headers(), timeout=60)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"RunPod status failed {status_url}: {e}")
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
    """Smart caller that supports both /runsync and /run."""
    e = endpoint.rstrip("/")
    if e.endswith("/runsync"):
        return _post_sync(endpoint, payload)
    if e.endswith("/run"):
        return _post_async(endpoint, payload)
    # fallback: try sync
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

# ---------- Exec worker (base64 in → base64 out) ----------
def call_comfy_exec(
    img_b64: str,
    mask_b64: str,
    prompt: str,
    *,
    steps: int = 22,
    cfg: float = 7.5,
    seed: int | None = None,
    invert_mask: bool = True,   # our rembg alpha usually has subject opaque; invert to inpaint background
) -> List[str]:
    payload = {
        "input": {
            "image_b64": img_b64,
            "mask_b64":  mask_b64,
            "prompt":    prompt,
            "steps":     steps,
            "cfg":       cfg,
            "invert_mask": invert_mask
        }
    }
    if seed is not None:
        payload["input"]["seed"] = int(seed)

    data = call_runpod(COMFY_ENDPOINT, payload)
    out = data.get("output") or {}

    images = out.get("images", [])
    if images and isinstance(images[0], dict) and "image" in images[0]:
        images = [im["image"] for im in images]
    if not images:
        raise HTTPException(status_code=502, detail=f"Exec worker returned no images: {data}")
    return images

# -------------- Routes --------------
@app.get("/debug/env")
def debug_env():
    return {
        "has_api_key": bool(RUNPOD_API_KEY),
        "mask_endpoint": MASK_ENDPOINT,
        "comfy_endpoint": COMFY_ENDPOINT,
        "comfy_mode": "runsync" if COMFY_ENDPOINT and COMFY_ENDPOINT.rstrip("/").endswith("runsync") else "run/async"
    }

@app.get("/styles", summary="List preset style keys")
def styles():
    return {"default": DEFAULT_STYLE_KEY, "presets": STYLE_PRESETS}

@app.post("/mask/test", summary="Test mask worker (returns mask only)")
async def mask_test(file: UploadFile = File(...), mode: str = Form("preview")):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    # best for edges: mask worker gets PNG of original
    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    return {"mode": mode, "mask_b64": mask_b64}

@app.post(
    "/preview/json",
    summary="Preview mockups (JSON, 5 variations) — set 'style' only"
)
async def preview_json(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY, description="Preset key (e.g. 'minimal_gallery') or free-text style"),
    mode: str = Form("preview", description="'preview' (fast) or 'final' (crisper mask)")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)

    # Mask: PNG of original; Exec: JPEG (or PNG) of clamped image
    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_exec = b64_jpeg(img_small, q=90)

    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for s in seeds:
        imgs = call_comfy_exec(
            img_b64_for_exec, mask_b64, prompt,
            steps=22, cfg=7.5, seed=s, invert_mask=True
        )
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})

    return {"mode": mode, "style": style, "prompt_used": prompt, "count": len(results), "results": results}

@app.post(
    "/preview/html",
    summary="Preview mockups (HTML gallery, 5 variations) — set 'style' only",
    response_class=HTMLResponse
)
async def preview_html(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY, description="Preset key (e.g. 'minimal_gallery') or free-text style"),
    mode: str = Form("preview", description="'preview' (fast) or 'final' (crisper mask)")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)

    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_exec = b64_jpeg(img_small, q=90)

    seeds = [secrets.randbits(32) for _ in range(5)]
    imgs64: List[str] = []
    for s in seeds:
        out = call_comfy_exec(
            img_b64_for_exec, mask_b64, prompt,
            steps=22, cfg=7.5, seed=s, invert_mask=True
        )
        if out:
            imgs64.append(out[0])

    html = "<h1>Generated Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for img64 in imgs64:
        html += f"<figure style='margin:8px'><img src='data:image/png;base64,{img64}' style='width:300px;margin:5px;display:block'/></figure>"
    html += "</div>"
    return HTMLResponse(content=html)
