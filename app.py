# app.py
import io
import os
import json
import time
import base64
import secrets
import zipfile
from typing import Dict, Any, List
from datetime import datetime, timedelta
from threading import Lock

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Mockup API",
    description="Upload an image and generate wall-art mockups via RunPod (mask worker + ComfyUI).",
    version="2.6"
)

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT    = os.getenv("RUNPOD_MASK_ENDPOINT")      # .../<MASK_ID>/runsync (recommended)
COMFY_ENDPOINT   = os.getenv("RUNPOD_COMFY_ENDPOINT")     # .../<COMFY_ID>/run  or .../runsync
COMFY_MODEL      = os.getenv("RUNPOD_COMFY_MODEL", "v1-5-pruned-emaonly.safetensors")  # change in Render as needed

# Warm-up config (to avoid cold-start delays while keeping Min Workers = 0)
WARMUP_ENABLED = os.getenv("WARMUP_ENABLED", "true").lower() in ("1", "true", "yes")
WARMUP_TTL_MIN = int(os.getenv("WARMUP_TTL_MIN", "15"))          # how long a warm stays "fresh"
WARMUP_TIMEOUT = int(os.getenv("WARMUP_TIMEOUT", "60"))          # seconds to allow a warm job

def _assert_env():
    missing = []
    if not RUNPOD_API_KEY: missing.append("RUNPOD_API_KEY")
    if not MASK_ENDPOINT:  missing.append("RUNPOD_MASK_ENDPOINT")
    if not COMFY_ENDPOINT: missing.append("RUNPOD_COMFY_ENDPOINT")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env: {', '.join(missing)}")

# -----------------------------------------------------------------------------
# Style presets
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def clamp_image(img: Image.Image, max_side=1280) -> Image.Image:
    """Scale down the larger side to max_side (keeps aspect)."""
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
    """
    Decode -> sanitize -> re-encode mask as clean RGBA PNG with alpha channel.
    If anything fails, returns the original b64 so we at least see Comfy's error.
    """
    try:
        raw = base64.b64decode(mask_b64)
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        alpha = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
        clean = Image.new("RGBA", im.size, (255, 255, 255, 0))
        clean.putalpha(alpha)
        buf = io.BytesIO()
        clean.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return mask_b64

# -----------------------------------------------------------------------------
# RunPod HTTP helpers
# -----------------------------------------------------------------------------
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

    # poll status
    base = endpoint_run.rsplit("/", 1)[0]
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
    e = endpoint.rstrip("/")
    if e.endswith("/runsync"):
        return _post_sync(endpoint, payload)
    if e.endswith("/run"):
        return _post_async(endpoint, payload)
    # fallback: try sync first
    return _post_sync(endpoint, payload)

# -----------------------------------------------------------------------------
# Warm-up helpers (to avoid cold-start latency/cost with Min Workers = 0)
# -----------------------------------------------------------------------------
_last_warm: Dict[str, datetime] = {}
_warm_lock = Lock()

def _tiny_png_b64() -> str:
    """1x1 transparent PNG as base64 (very small, cheap)."""
    buf = io.BytesIO()
    Image.new("RGBA", (1, 1), (255, 255, 255, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _needs_warm(key: str) -> bool:
    if not WARMUP_ENABLED:
        return False
    with _warm_lock:
        last = _last_warm.get(key)
    if not last:
        return True
    return datetime.utcnow() - last > timedelta(minutes=WARMUP_TTL_MIN)

def _mark_warm(key: str) -> None:
    with _warm_lock:
        _last_warm[key] = datetime.utcnow()

def _post_sync_timeout(endpoint: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    try:
        r = requests.post(endpoint, headers=_headers(), json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
    return r.json()

def _warm_mask():
    tiny = _tiny_png_b64()
    payload = {"input": {"image_b64": tiny, "mode": "preview"}}
    e = MASK_ENDPOINT.rstrip("/")
    if e.endswith("/runsync"):
        _post_sync_timeout(MASK_ENDPOINT, payload, timeout=WARMUP_TIMEOUT)
    else:
        _post_async(MASK_ENDPOINT, payload, timeout_s=WARMUP_TIMEOUT)

def _warm_comfy():
    tiny = _tiny_png_b64()
    # Workflow that does not load a checkpoint; just load & save an image.
    wf = {"img": {"class_type": "LoadImage", "inputs": {"image": tiny}},
          "out": {"class_type": "SaveImage", "inputs": {"images": ["img", 0]}}}
    payload = {"input": {"return_type": "base64", "workflow": wf}}
    e = COMFY_ENDPOINT.rstrip("/")
    if e.endswith("/runsync"):
        _post_sync_timeout(COMFY_ENDPOINT, payload, timeout=WARMUP_TIMEOUT)
    else:
        _post_async(COMFY_ENDPOINT, payload, timeout_s=WARMUP_TIMEOUT)

def call_runpod_warm(endpoint: str, payload: Dict[str, Any], warm_key: str) -> Dict[str, Any]:
    """
    Wrapper that warms the target once per TTL, then sends the real payload.
    Retries once after warm on failure (helpful for cold starts).
    warm_key: "mask" or "comfy"
    """
    # Warm if needed
    if _needs_warm(warm_key):
        try:
            if warm_key == "mask":
                _warm_mask()
            else:
                _warm_comfy()
            _mark_warm(warm_key)
        except Exception:
            # warming failure shouldn't block; continue and try real call anyway
            pass

    # Try the real call
    try:
        return call_runpod(endpoint, payload)
    except HTTPException:
        # Retry once after a forced warm
        try:
            if warm_key == "mask":
                _warm_mask()
            else:
                _warm_comfy()
            _mark_warm(warm_key)
        except Exception:
            pass
        return call_runpod(endpoint, payload)

# -----------------------------------------------------------------------------
# Mask worker
# -----------------------------------------------------------------------------
def call_mask_worker(img_b64: str, mode: str = "preview") -> str:
    payload = {"input": {"image_b64": img_b64, "mode": mode}}
    data = call_runpod_warm(MASK_ENDPOINT, payload, warm_key="mask")
    out = data.get("output") or {}
    mask_b64 = out.get("mask_b64") or data.get("mask_b64")
    if not mask_b64:
        raise HTTPException(status_code=502, detail=f"Mask worker returned no mask: {data}")
    return normalize_mask_b64(mask_b64)

# -----------------------------------------------------------------------------
# ComfyUI inpaint workflow
# -----------------------------------------------------------------------------
def comfy_workflow(seed: int, prompt: str) -> Dict[str, Any]:
    """
    Vanilla Comfy inpaint pipeline:
      - CheckpointLoaderSimple -> model/clip/vae
      - LoadImage + LoadImageMask
      - VAEEncodeForInpaint(image, mask) -> latent + mask
      - KSampler(model, pos/neg, latent, noise, mask)
      - VAEDecode -> SaveImage
    """
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
    data = call_runpod_warm(COMFY_ENDPOINT, payload, warm_key="comfy")
    out = data.get("output") or {}
    images = out.get("images", [])
    if not images and isinstance(out, list):
        images = [{"image": x} for x in out]
    if not images:
        raise HTTPException(status_code=502, detail=f"Comfy returned no images: {data}")
    return [e["image"] for e in images if isinstance(e, dict) and "image" in e]

# -----------------------------------------------------------------------------
# ZIP helper
# -----------------------------------------------------------------------------
def _zip_bytes_from_images(items: List[dict], style_label: str = "mockup") -> bytes:
    """
    items: list of {"seed": int, "image_b64": "<base64 png>"} records
    Returns: raw bytes of a zip file containing all images as PNG.
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = [{"seed": it["seed"], "filename": f"{style_label}_{it['seed']}.png"} for it in items]
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        for it in items:
            filename = f"{style_label}_{it['seed']}.png"
            try:
                png_bytes = base64.b64decode(it["image_b64"])
            except Exception:
                continue
            zf.writestr(filename, png_bytes)

    mem.seek(0)
    return mem.read()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/debug/env")
def debug_env():
    return {
        "has_api_key": bool(RUNPOD_API_KEY),
        "mask_endpoint": MASK_ENDPOINT,
        "comfy_endpoint": COMFY_ENDPOINT,
        "comfy_model": COMFY_MODEL,
        "warmup_enabled": WARMUP_ENABLED,
        "warmup_ttl_min": WARMUP_TTL_MIN
    }

@app.post("/debug/warm")
def debug_warm():
    """
    Manually warm both endpoints now (useful before a traffic spike).
    """
    out = {"ok": True, "ttl_min": WARMUP_TTL_MIN, "warmed": []}
    try:
        _warm_mask()
        _mark_warm("mask")
        out["warmed"].append("mask")
    except Exception as e:
        out["ok"] = False
        out["mask_error"] = str(e)
    try:
        _warm_comfy()
        _mark_warm("comfy")
        out["warmed"].append("comfy")
    except Exception as e:
        out["ok"] = False
        out["comfy_error"] = str(e)
    return out

@app.get("/styles", summary="List preset style keys")
def styles():
    return {"default": DEFAULT_STYLE_KEY, "presets": STYLE_PRESETS}

@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    # minimal demo page with HTML preview + ZIP form
    style_options = "".join(
        f'<option value="{k}" {"selected" if k==DEFAULT_STYLE_KEY else ""}>{k}</option>'
        for k in STYLE_PRESETS.keys()
    )
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Mockup Demo</title>
      <style>
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 24px; }}
        .card {{ max-width: 720px; background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.06); }}
        h1 {{ margin: 0 0 16px 0; font-size: 20px; }}
        .row {{ margin-bottom: 14px; }}
        input[type=file], select {{ padding: 8px; width: 100%; max-width: 360px; }}
        button {{ background:#111827; color:#fff; padding:10px 16px; border-radius:10px; border:none; cursor:pointer; }}
        button:hover {{ background:#0b1220; }}
        hr {{ border:none; border-top:1px solid #eee; margin: 24px 0; }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Preview in browser</h1>
        <form method="post" enctype="multipart/form-data" action="/preview/html" target="_blank">
          <div class="row"><input type="file" name="file" required></div>
          <div class="row">
            <label>Style:&nbsp;</label>
            <select name="style">{style_options}</select>
          </div>
          <div class="row">
            <label>Mode:&nbsp;</label>
            <select name="mode">
              <option value="preview" selected>preview</option>
              <option value="final">final</option>
            </select>
          </div>
          <button type="submit">Generate HTML Preview</button>
        </form>

        <hr />

        <h1>Download ZIP of images</h1>
        <form method="post" enctype="multipart/form-data" action="/preview/zip">
          <div class="row"><input type="file" name="file" required></div>
          <div class="row">
            <label>Style:&nbsp;</label>
            <select name="style">{style_options}</select>
          </div>
          <div class="row">
            <label>Mode:&nbsp;</label>
            <select name="mode">
              <option value="preview" selected>preview</option>
              <option value="final">final</option>
            </select>
          </div>
          <button type="submit">Generate ZIP</button>
        </form>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/mask/test", summary="Test mask worker (returns mask only)")
async def mask_test(file: UploadFile = File(...), mode: str = Form("preview")):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    return {"mode": mode, "mask_b64": mask_b64}

@app.post("/preview/json", summary="Preview mockups (JSON, 5 variations) — set 'style' and 'mode'")
async def preview_json(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY, description="Preset key (e.g. 'minimal_gallery') or free-text style"),
    mode: str = Form("preview", description="'preview' (fast) or 'final' (crisper mask)")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)

    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_comfy = b64_jpeg(img_small, q=90)

    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for s in seeds:
        imgs = call_comfy(img_b64_for_comfy, mask_b64, prompt, seed=s)
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})

    return {"mode": mode, "style": style, "prompt_used": prompt, "count": len(results), "results": results}

@app.post("/preview/html", summary="Preview mockups (HTML gallery, 5 variations)", response_class=HTMLResponse)
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

@app.post("/preview/zip", summary="Preview mockups (ZIP with 5 PNGs) — set 'style' and 'mode'")
async def preview_zip(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY, description="Preset key (e.g. 'minimal_gallery') or free-text style"),
    mode: str = Form("preview", description="'preview' (fast) or 'final' (crisper mask)")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)
    style_label = style if style in STYLE_PRESETS else "custom"

    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_comfy = b64_jpeg(img_small, q=90)

    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for s in seeds:
        imgs = call_comfy(img_b64_for_comfy, mask_b64, prompt, seed=s)
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})

    if not results:
        raise HTTPException(status_code=502, detail="No images returned; try different settings.")

    zip_bytes = _zip_bytes_from_images(results, style_label=style_label)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{style_label}_mockups.zip"'}
    )
