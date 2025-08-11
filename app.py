# app.py — Mockup API (Preview + Finalise)

import io
import os
import json
import base64
import secrets
import subprocess
import sys
from typing import List, Tuple, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from rembg import remove, new_session
import requests

# ----------------------------
# Environment configuration
# ----------------------------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "")  # e.g. https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync

# Preview behaviour (low RAM)
MAX_PREVIEW_SIDE = int(os.getenv("MAX_PREVIEW_SIDE", "1536"))  # long side clamp
PREVIEW_MODEL = os.getenv("PREVIEW_MODEL", "u2netp")  # smallest model available in rembg==2.0.67

# Finalise behaviour (higher quality)
MAX_FINAL_SIDE = int(os.getenv("MAX_FINAL_SIDE", "2048"))
FINALISE_MODEL = os.getenv("FINALISE_MODEL", "u2net")  # higher quality

# Upload limits
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))   # 8 MB
MAX_PIXELS = int(os.getenv("MAX_UPLOAD_PIXELS", str(12_000_000)))      # safety net

# Concurrency knobs for small instances
os.environ.setdefault("WEB_CONCURRENCY", "1")
os.environ.setdefault("UVICORN_WORKERS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ----------------------------
# App & CORS
# ----------------------------
app = FastAPI(title="Mockup API (Preview + Finalise)", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# rembg sessions (lazy init to avoid build failures)
# ----------------------------
_PREVIEW_SESSION = None
_FINALISE_SESSION = None

def get_preview_session():
    global _PREVIEW_SESSION
    if _PREVIEW_SESSION is None:
        _PREVIEW_SESSION = _safe_new_session(PREVIEW_MODEL, ["u2netp", "u2net"])
    return _PREVIEW_SESSION

def get_finalise_session():
    global _FINALISE_SESSION
    if _FINALISE_SESSION is None:
        _FINALISE_SESSION = _safe_new_session(FINALISE_MODEL, ["u2net", "u2netp"])
    return _FINALISE_SESSION

def _safe_new_session(name: str, fallbacks: list[str]):
    try:
        return new_session(name)
    except Exception:
        for fb in fallbacks:
            try:
                return new_session(fb)
            except Exception:
                continue
        raise

# ----------------------------
# Utilities
# ----------------------------
def ensure_env():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
        raise HTTPException(status_code=500, detail="RunPod environment not configured")

def pil_to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def clamp_image(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def read_upload_image(file: UploadFile) -> Image.Image:
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Image too large")
    return img

def make_mask(img_rgba: Image.Image, *, mode: str) -> Image.Image:
    """
    mode = 'preview' or 'final'
    - preview: low RAM, alpha_matting=False using u2netp by default
    - final: higher quality, alpha_matting=True using u2net
    """
    session = get_preview_session() if mode == "preview" else get_finalise_session()
    alpha_matting = False if mode == "preview" else True

    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG")

    out_bytes = remove(
        buf.getvalue(),
        session=session,
        alpha_matting=alpha_matting,
        post_process_mask=True,  # cleaner edges without high RAM cost
    )
    return Image.open(io.BytesIO(out_bytes)).convert("RGBA")

def build_workflow(seed: int, prompt: str) -> dict:
    """
    Replace with your exact ComfyUI graph if needed.
    We pass image/mask via placeholders and keep mask channel 'alpha' per Golden Rules.
    """
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": "__b64_img__"}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": "__b64_mask__", "channel": "alpha"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "noise": {"class_type": "RandomNoise", "inputs": {"seed": seed}},
        "prep": {"class_type": "ApplyMaskToLatent", "inputs": {"image": ["img", 0], "mask": ["mask", 0]}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["checkpoint", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["prep", 0],
                "noise": ["noise", 0],
                "steps": 22,
                "cfg": 7.5,
                "sampler_name": "euler"
            }
        },
        # Optional: add tiny feather/contact shadow in your own graph here
        "out": {"class_type": "SaveImage", "inputs": {"images": ["sampler", 0]}}
    }

def call_runpod(workflow: dict, img_b64: str, mask_b64: str, timeout: int = 120) -> List[str]:
    ensure_env()
    # substitute placeholders
    wf_json = json.dumps(workflow).replace("__b64_img__", img_b64).replace("__b64_mask__", mask_b64)
    workflow_payload = json.loads(wf_json)

    payload = {"input": {"return_type": "base64", "workflow": workflow_payload}}
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    try:
        resp = requests.post(RUNPOD_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error calling RunPod: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"RunPod error: {resp.text}")

    data = resp.json()
    images = data.get("output", {}).get("images", [])
    if not images:
        images = [{"image": im} for im in data.get("output", [])]  # some templates return list[str]
    b64_list = [im.get("image") for im in images if isinstance(im, dict) and "image" in im]
    if not b64_list:
        raise HTTPException(status_code=502, detail="RunPod returned no images")
    return b64_list

def html_gallery(items: List[Tuple[str, int]], title: str) -> str:
    imgs = "\n".join(
        f"<figure style='margin:8px'><img style='width:300px;height:auto;display:block' src='data:image/png;base64,{b64}'/>"
        f"<figcaption style='font:12px system-ui;color:#444'>seed {seed}</figcaption></figure>"
        for b64, seed in items
    )
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title></head>
<body style="margin:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto">
  <h1 style="margin:0 0 12px">{title}</h1>
  <div style="display:flex;flex-wrap:wrap">{imgs}</div>
</body>
</html>"""

# ----------------------------
# Middleware: simple size gate
# ----------------------------
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_BYTES:
        return JSONResponse({"detail": "File too large"}, status_code=413)
    return await call_next(request)

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
<body style="margin:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto">
  <h1>Mockup Preview + Finalise</h1>
  <form action="/preview/html" method="post" enctype="multipart/form-data" style="display:grid;gap:8px;max-width:520px">
    <input type="file" name="file" accept="image/*" required />
    <input type="text" name="prompt" placeholder="Describe the background…" required />
    <button type="submit">Generate 5 preview variations</button>
  </form>
  <p style="color:#666">Programmatic: POST /preview/json or /finalise/json</p>
</body>
</html>
""".strip()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/debug/deps")
def debug_deps():
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "list"], text=True)
    except Exception as e:
        out = f"pip list failed: {e}"
    return {"python": sys.version, "pip_list": out}

# ---------- PREVIEW (RAM-friendly) ----------

@app.post("/preview/json")
async def preview_json(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    # Read and clamp
    try:
        original = read_upload_image(file)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    original = clamp_image(original, MAX_PREVIEW_SIDE)

    # Mask (lite)
    try:
        mask = make_mask(original, mode="preview")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mask generation failed: {e}")

    img_b64 = pil_to_png_b64(original)
    mask_b64 = pil_to_png_b64(mask)

    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for seed in seeds:
        wf = build_workflow(seed=seed, prompt=prompt)
        imgs = call_runpod(wf, img_b64=img_b64, mask_b64=mask_b64)
        results.append({"seed": seed, "image_b64": imgs[0]})

    return JSONResponse({"count": len(results), "results": results, "mode": "preview"})

@app.post("/preview/html", response_class=HTMLResponse)
async def preview_html(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    data = await preview_json(file=file, prompt=prompt)
    payload = json.loads(data.body.decode("utf-8"))
    items = [(r["image_b64"], r["seed"]) for r in payload["results"]]
    return HTMLResponse(content=html_gallery(items, "Preview mockups (5)"))

# ---------- FINALISE (higher quality) ----------

@app.post("/finalise/json")
async def finalise_json(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    seed: Optional[int] = Form(None),
    target_side: Optional[int] = Form(None)  # override if you want e.g. 3000
):
    """
    Re-run one variation at higher quality.
    - Provide the same image and prompt as preview.
    - Optionally pass 'seed' from a preview result to reproduce composition.
    - Uses a stronger rembg session and alpha_matting=True.
    """
    try:
        original = read_upload_image(file)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    side = int(target_side) if target_side else MAX_FINAL_SIDE
    original = clamp_image(original, side)

    try:
        mask = make_mask(original, mode="final")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mask generation failed: {e}")

    img_b64 = pil_to_png_b64(original)
    mask_b64 = pil_to_png_b64(mask)

    chosen_seed = int(seed) if seed is not None else secrets.randbits(32)
    wf = build_workflow(seed=chosen_seed, prompt=prompt)
    imgs = call_runpod(wf, img_b64=img_b64, mask_b64=mask_b64)

    return JSONResponse({
        "seed": chosen_seed,
        "image_b64": imgs[0],
        "mode": "final",
        "side": side,
        "model": FINALISE_MODEL
    })

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
