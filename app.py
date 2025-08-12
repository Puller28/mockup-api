# app.py
import io
import os
import json
import time
import base64
import secrets
from typing import Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI(
    title="Mockup API",
    description="Upload an image and generate wall-art mockups via RunPod (mask worker + ComfyUI).",
    version="2.5"
)

# ---------------- Env ----------------
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT    = os.getenv("RUNPOD_MASK_ENDPOINT")      # .../<MASK_ID>/runsync
COMFY_ENDPOINT   = os.getenv("RUNPOD_COMFY_ENDPOINT")     # .../<COMFY_ID>/run  (async) or .../runsync
# Default to Flux checkpoint because that’s what your endpoint showed; change if needed.
COMFY_MODEL      = os.getenv("RUNPOD_COMFY_MODEL", "flux1-dev-fp8.safetensors")

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
    """Downscale longest side to max_side while keeping aspect ratio."""
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

def resize_mask_b64(mask_b64: str, target_size: Tuple[int, int]) -> str:
    """
    Decode mask -> convert to RGBA -> resize to target_size with NEAREST
    (keeps crisp edges) -> re-encode as clean PNG base64.
    """
    try:
        raw = base64.b64decode(mask_b64)
        im = Image.open(io.BytesIO(raw)).convert("RGBA")

        # Build alpha channel
        alpha = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
        alpha_resized = alpha.resize(target_size, Image.NEAREST)

        clean = Image.new("RGBA", target_size, (255, 255, 255, 0))
        clean.putalpha(alpha_resized)

        buf = io.BytesIO()
        clean.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # Fall back to original if anything odd happens
        return mask_b64

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
    """
    Robust inpaint graph (vanilla-safe):
      - CheckpointLoaderSimple -> model/clip/vae
      - LoadImage (subject) + LoadImage (mask) -> ImageToMask
      - VAEEncodeForInpaint(pixels, mask)
      - KSampler(model, pos/neg, latent, noise, mask)
      - VAEDecode -> SaveImage
    """
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": COMFY_MODEL}},

        # Inputs
        "img":      {"class_type": "LoadImage", "inputs": {"image": "__b64_img__"}},
        "mask_img": {"class_type": "LoadImage", "inputs": {"image": "__b64_mask__"}},

        # Convert image -> mask tensor (most tolerant)
        "to_mask":  {"class_type": "ImageToMask", "inputs": {"image": ["mask_img", 0]}},

        # Text encodes
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "",      "clip": ["ckpt", 1]}},

        # Latent encode for inpaint
        "enc": {"class_type": "VAEEncodeForInpaint",
                "inputs": {"pixels": ["img", 0], "mask": ["to_mask", 0], "vae": ["ckpt", 2]}},

        # Sample
        "noise": {"class_type": "RandomNoise", "inputs": {"seed": seed}},
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": ["ckpt", 0],
                               "positive": ["pos", 0],
                               "negative": ["neg", 0],
                               "latent_image": ["enc", 0],
                               "noise": ["noise", 0],
                               "mask": ["enc", 1],
                               "steps": 22,
                               "cfg": 7.5,
                               "sampler_name": "euler"}},

        # Decode & save
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
    # ask worker on original for best edges
    mask_b64_raw = call_mask_worker(b64_png(img), mode=mode)
    # resize & sanitize to a manageable preview size for a quick check
    img_small = clamp_image(img, max_side=512)
    mask_b64 = resize_mask_b64(mask_b64_raw, img_small.size)
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

    # 1) Prepare preview image first (so we know target size for mask)
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_comfy = b64_png(img_small)  # PNG for robustness (can switch to JPEG later)

    # 2) Mask from worker on full image, then resize/sanitize to preview size
    mask_b64_raw = call_mask_worker(b64_png(img), mode=mode)
    mask_b64 = resize_mask_b64(mask_b64_raw, img_small.size)

    # 3) Generate 5 variations
    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for s in seeds:
        imgs = call_comfy(img_b64_for_comfy, mask_b64, prompt, seed=s)
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})

    return {"mode": mode, "style": style, "prompt_used": prompt, "count": len(results), "results": results}

from fastapi import Query

def _tiny_png_b64(size=64, invert=False) -> str:
    # 64x64 checkerboard to avoid “empty” mask edge cases
    im = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    px = im.load()
    for y in range(size):
        for x in range(size):
            v = 255 if ((x//8 + y//8) % 2 == (0 if not invert else 1)) else 0
            px[x, y] = (v, v, v, 255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _wf_variant(variant: str, model_name: str, img_b64: str, mask_b64: str):
    """
    Build 3 workflow variants to discover the exact input contract Comfy expects.
    - A: LoadImage(image="<base64>")               -> VAEEncodeForInpaint (mask via ImageToMask)
    - B: LoadImage(image={"image": "<base64>"})    -> VAEEncodeForInpaint (mask via ImageToMask)
    - C: Same as A but force mask path explicitly
    """
    # common backbone
    ckpt = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model_name}}
    pos  = {"class_type": "CLIPTextEncode", "inputs": {"text": "test", "clip": ["ckpt", 1]}}
    neg  = {"class_type": "CLIPTextEncode", "inputs": {"text": "",     "clip": ["ckpt", 1]}}
    noise = {"class_type": "RandomNoise", "inputs": {"seed": 12345}}

    if variant.upper() == "A":
        img      = {"class_type": "LoadImage", "inputs": {"image": img_b64}}
        mask_img = {"class_type": "LoadImage", "inputs": {"image": mask_b64}}

    elif variant.upper() == "B":
        img      = {"class_type": "LoadImage", "inputs": {"image": {"image": img_b64}}}
        mask_img = {"class_type": "LoadImage", "inputs": {"image": {"image": mask_b64}}}

    else:  # "C"
        img      = {"class_type": "LoadImage", "inputs": {"image": img_b64}}
        mask_img = {"class_type": "LoadImage", "inputs": {"image": mask_b64}}

    wf = {
        "ckpt": ckpt,
        "img": img,
        "mask_img": mask_img,
        "to_mask": {"class_type": "ImageToMask", "inputs": {"image": ["mask_img", 0]}},
        "pos": pos,
        "neg": neg,
        "enc": {"class_type": "VAEEncodeForInpaint",
                "inputs": {"pixels": ["img", 0], "mask": ["to_mask", 0], "vae": ["ckpt", 2]}},
        "noise": noise,
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": ["ckpt", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["enc", 0], "noise": ["noise", 0],
                               "mask": ["enc", 1], "steps": 5, "cfg": 4.0, "sampler_name": "euler"}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]}},
        "out": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}}
    }
    return wf

@app.post("/debug/comfy-sanity")
def debug_comfy_sanity(
    variant: str = Query("A", description="A|B|C input shape test"),
    model: str = Query(COMFY_MODEL, description="Checkpoint filename on the Comfy endpoint")
):
    _assert_env()
    # create tiny test image & mask
    img_b64  = _tiny_png_b64(64, invert=False)
    mask_b64 = _tiny_png_b64(64, invert=True)

    wf = _wf_variant(variant, model, img_b64, mask_b64)
    payload = {"input": {"return_type": "base64", "workflow": wf}}

    try:
        data = call_runpod(COMFY_ENDPOINT, payload)
    except HTTPException as e:
        # return full error detail to us without swallowing it
        return {"variant": variant, "error": True, "detail": e.detail}

    return {
        "variant": variant,
        "status": data.get("status") or "OK",
        "has_output_images": bool((data.get("output") or {}).get("images")),
        "raw": data
    }


@app.post("/preview/html", response_class=HTMLResponse)
async def preview_html(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY),
    mode: str = Form("preview")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    prompt = build_prompt(style)

    img_small = clamp_image(img, max_side=1280)
    img_b64_for_comfy = b64_png(img_small)

    mask_b64_raw = call_mask_worker(b64_png(img), mode=mode)
    mask_b64 = resize_mask_b64(mask_b64_raw, img_small.size)

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
