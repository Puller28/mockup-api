# app.py — Mask worker + Comfy/Exec render, refined edges, local ZIP

import io, os, json, time, base64, secrets, zipfile
from typing import Dict, Any, List, Tuple, Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from PIL import Image

app = FastAPI(
    title="Mockup API",
    description="Mask worker + Comfy/Exec render, with local ZIP bundling and debug tools.",
    version="3.0"
)

# ---------------- Env ----------------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# Mask worker (RunPod serverless Python)
RUNPOD_MASK_ENDPOINT  = os.getenv("RUNPOD_MASK_ENDPOINT")         # .../<MASK_ID>/runsync

# Comfy or Exec renderer — choose one
RUNPOD_COMFY_ENDPOINT = os.getenv("RUNPOD_COMFY_ENDPOINT")        # .../<COMFY_ID>/run  (async)
RUNPOD_EXEC_ENDPOINT  = os.getenv("RUNPOD_EXEC_ENDPOINT")         # .../<EXEC_ID>/run or /runsync
USE_EXEC              = os.getenv("USE_EXEC", "false").lower() == "true"

# Comfy checkpoint
COMFY_MODEL           = os.getenv("RUNPOD_COMFY_MODEL", "v1-5-pruned-emaonly.safetensors")

# Warmup
WARMUP_ENABLED        = os.getenv("WARMUP_ENABLED", "false").lower() == "true"

def _assert_env():
    missing = []
    if not RUNPOD_API_KEY:         missing.append("RUNPOD_API_KEY")
    if not RUNPOD_MASK_ENDPOINT:   missing.append("RUNPOD_MASK_ENDPOINT")
    if USE_EXEC:
        if not RUNPOD_EXEC_ENDPOINT:
            missing.append("RUNPOD_EXEC_ENDPOINT")
    else:
        if not RUNPOD_COMFY_ENDPOINT:
            missing.append("RUNPOD_COMFY_ENDPOINT")
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

def build_prompt(style_value: Optional[str]) -> str:
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
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_jpeg(img: Image.Image, q=90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------- HTTP helpers ---------------
def _headers_json() -> Dict[str, str]:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def call_runpod(endpoint: str, payload: dict | str) -> dict:
    """POST JSON safely to RunPod; supports already-encoded JSON strings."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            r = requests.post(endpoint, headers=_headers_json(), data=payload, timeout=180)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
            return r.json()

    r = requests.post(endpoint, headers=_headers_json(), json=payload, timeout=180)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
    return r.json()

def runpod_run_and_wait(endpoint_run: str, payload: dict, timeout_s: int = 240, poll_every: float = 1.5) -> dict:
    """POST to /run, then poll /status/{id} until COMPLETED/FAILED."""
    r = requests.post(endpoint_run, headers=_headers_json(), json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint_run}: {r.text}")
    job = r.json()
    job_id = job.get("id") or job.get("output", {}).get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod /run did not return a job id: {job}")

    base = endpoint_run.rstrip("/").rsplit("/", 1)[0]  # strip "/run"
    status_url = f"{base}/status/{job_id}"
    started = time.time()

    while True:
        s = requests.get(status_url, headers=_headers_json(), timeout=30)
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

# ---------- Mask worker ----------
def call_mask_worker(
    mode: str,
    img_b64: str,
    timeout: int = 60,
    feather_px: float = 1.0,
    decontaminate: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Returns (mask_b64, cutout_b64). mask_b64 is required, cutout_b64 may be None.
    """
    _assert_env()
    payload = {
        "input": {
            "image_b64": img_b64,
            "mode": mode,                      # "preview" or "final"
            "feather_px": feather_px,          # 1.0–1.5 is a good range
            "decontaminate": decontaminate
        }
    }
    resp = requests.post(RUNPOD_MASK_ENDPOINT, headers=_headers_json(), json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Mask worker error: {resp.text}")
    data = resp.json()
    out = data.get("output") or {}
    mask_b64 = out.get("mask_b64") or data.get("mask_b64")
    cutout_b64 = out.get("cutout_b64") or data.get("cutout_b64")
    if not mask_b64:
        raise HTTPException(status_code=502, detail="Mask worker returned no mask_b64")
    return mask_b64, cutout_b64

# ---------- ComfyUI workflow ----------
def comfy_workflow(seed: int, prompt: str) -> Dict[str, Any]:
    """
    Stock inpaint pipeline:
      CheckpointLoaderSimple -> model/clip/vae
      LoadImage + LoadImageMask (base64 objects)
      VAEEncodeForInpaint -> KSampler -> VAEDecode -> SaveImage
    """
    return {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": COMFY_MODEL}
        },
        "img": {
            "class_type": "LoadImage",
            "inputs": {"image": {"image": "__b64_img__", "type": "base64"}}
        },
        "mask": {
            "class_type": "LoadImageMask",
            "inputs": {"image": {"image": "__b64_mask__", "type": "base64"}, "channel": "alpha"}
        },
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["ckpt", 1]}},
        "enc": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["img", 0], "mask": ["mask", 0], "vae": ["ckpt", 2]}},
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
        "out": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}}
    }

def call_comfy(img_b64_raw: str, mask_b64_raw: str, prompt: str, seed: int) -> List[str]:
    """
    img_b64_raw and mask_b64_raw must be raw base64 strings (no data URLs).
    """
    wf = json.loads(
        json.dumps(comfy_workflow(seed, prompt))
        .replace("__b64_img__", img_b64_raw)
        .replace("__b64_mask__", mask_b64_raw)
    )
    payload = {"input": {"return_type": "base64", "workflow": wf}}
    data = runpod_run_and_wait(RUNPOD_COMFY_ENDPOINT, payload)
    out = data.get("output") or {}
    images = out.get("images", [])
    if not images and isinstance(out, list):
        images = [{"image": x} for x in out]
    if not images:
        raise HTTPException(status_code=502, detail=f"Comfy returned no images: {data}")
    return [e["image"] for e in images if isinstance(e, dict) and "image" in e]

# ---------- Exec worker (optional renderer) ----------
def call_exec_inpaint(img_b64_raw: str, mask_b64_raw: str, prompt: str, seed: int) -> List[str]:
    """
    Call an Exec worker that takes base64 directly and returns base64 images.
    Supports /runsync or /run endpoints.
    """
    if not RUNPOD_EXEC_ENDPOINT:
        raise HTTPException(status_code=500, detail="RUNPOD_EXEC_ENDPOINT not set")
    payload = {"input": {"image_b64": img_b64_raw, "mask_b64": mask_b64_raw, "prompt": prompt, "seed": seed, "return_type": "base64"}}
    ep = RUNPOD_EXEC_ENDPOINT.rstrip("/")
    if ep.endswith("/runsync"):
        data = call_runpod(ep, payload)
    else:
        data = runpod_run_and_wait(ep, payload)
    out = data.get("output") or {}
    images = out.get("images", [])
    if isinstance(out, list):
        images = out
    if not images:
        raise HTTPException(status_code=502, detail=f"Exec worker returned no images: {data}")
    result = []
    for e in images:
        if isinstance(e, dict) and "image" in e: result.append(e["image"])
        elif isinstance(e, str): result.append(e)
    return result

# -------------- Routes --------------
@app.get("/debug/env")
def debug_env():
    return {
        "has_api_key": bool(RUNPOD_API_KEY),
        "mask_endpoint": RUNPOD_MASK_ENDPOINT,
        "renderer_mode": "exec" if USE_EXEC else "comfy",
        "renderer_endpoint": RUNPOD_EXEC_ENDPOINT if USE_EXEC else RUNPOD_COMFY_ENDPOINT,
        "comfy_model": COMFY_MODEL,
        "warmup_enabled": WARMUP_ENABLED
    }

@app.get("/styles")
def styles():
    return {"default": DEFAULT_STYLE_KEY, "presets": STYLE_PRESETS}

@app.post("/mask/test")
async def mask_test(file: UploadFile = File(...), mode: str = Form("preview")):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    mask_b64, cutout_b64 = call_mask_worker(mode=mode, img_b64=b64_png(img))
    return {"mode": mode, "mask_b64": mask_b64, "cutout_b64": cutout_b64}

# keep last artefacts for quick download
_last_mask_png: bytes | None = None
_last_cutout_png: bytes | None = None

def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()
def _img_to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(img)).decode("utf-8")

def _make_checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    bg = Image.new("RGB", (w, h), (200, 200, 200)); px = bg.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (220,220,220) if ((x//tile)+(y//tile))%2==0 else (180,180,180)
    return bg

def _apply_mask_cutout(src_rgba: Image.Image, mask_rgba: Image.Image, invert: bool = False) -> tuple[Image.Image, Image.Image]:
    w, h = src_rgba.size
    alpha = mask_rgba.getchannel("A") if "A" in mask_rgba.getbands() else mask_rgba.convert("L")
    if invert:
        alpha = Image.eval(alpha, lambda a: 255 - a)
    bg_only = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg_only = Image.composite(src_rgba, bg_only, alpha)
    inv_alpha = Image.eval(alpha, lambda a: 255 - a)
    subj = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    subj = Image.composite(src_rgba, subj, inv_alpha)
    return subj, bg_only

@app.post("/debug/mask", response_class=HTMLResponse)
async def debug_mask(file: UploadFile = File(...), mode: str = Form("preview"), invert: int = Form(0)):
    _assert_env()
    src = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    w, h = src.size
    mask_b64, cutout_b64 = call_mask_worker(mode=mode, img_b64=b64_png(src), feather_px=1.2 if mode=="preview" else 1.5, decontaminate=(mode=="final"))
    mask = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("RGBA")
    cutout, bg_only = _apply_mask_cutout(src, mask, invert=bool(invert))
    # keep last outputs
    global _last_mask_png, _last_cutout_png
    _last_mask_png, _last_cutout_png = _png_bytes(mask), _png_bytes(cutout)
    checker = _make_checkerboard(w, h).convert("RGBA")
    white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    mask_over_checker = Image.composite(white, checker, mask.getchannel("A"))
    html = f"""
    <h1>Mask Debug</h1>
    <p><b>mode:</b> {mode} &nbsp;&nbsp; <b>invert:</b> {bool(invert)}</p>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <figure><figcaption>Original</figcaption><img style="max-width:380px" src="{_img_to_data_url(src)}"/></figure>
      <figure><figcaption>Raw Mask (PNG)</figcaption><img style="max-width:380px" src="{_img_to_data_url(mask)}"/></figure>
      <figure><figcaption>Mask on Checkerboard (alpha)</figcaption><img style="max-width:380px" src="{_img_to_data_url(mask_over_checker)}"/></figure>
      <figure><figcaption>Subject Cutout (kept area)</figcaption><img style="max-width:380px" src="{_img_to_data_url(cutout)}"/></figure>
      <figure><figcaption>Background-Only (to be inpainted)</figcaption><img style="max-width:380px" src="{_img_to_data_url(bg_only)}"/></figure>
    </div>
    <p><a href="/debug/last-mask.png" target="_blank">Download last mask</a> |
       <a href="/debug/last-cutout.png" target="_blank">Download last cutout</a></p>
    """
    return HTMLResponse(content=html)

@app.get("/debug/last-mask.png")
def get_last_mask():
    if _last_mask_png is None:
        return Response(status_code=404)
    return Response(content=_last_mask_png, media_type="image/png")

@app.get("/debug/last-cutout.png")
def get_last_cutout():
    if _last_cutout_png is None:
        return Response(status_code=404)
    return Response(content=_last_cutout_png, media_type="image/png")

# 1x1 transparent PNG
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="

@app.get("/warmup")
def warmup():
    if not WARMUP_ENABLED:
        return {"ok": True, "skipped": True}
    try:
        call_runpod(RUNPOD_MASK_ENDPOINT, {"input": {"image_b64": _TINY_PNG_B64, "mode": "preview"}})
    except Exception:
        pass
    try:
        ep = RUNPOD_EXEC_ENDPOINT if USE_EXEC else RUNPOD_COMFY_ENDPOINT
        if ep:
            payload = {"input": {"image_b64": _TINY_PNG_B64, "mask_b64": _TINY_PNG_B64, "prompt": "warmup", "seed": 1, "return_type": "base64"}}
            _ = call_runpod(ep, payload) if ep.endswith("/runsync") else runpod_run_and_wait(ep, payload)
    except Exception:
        pass
    return {"ok": True, "warmed": True}

# ---------- generation helpers ----------
def _render_once(img_b64_raw: str, mask_b64_raw: str, prompt: str, seed: int) -> List[str]:
    if USE_EXEC:
        return call_exec_inpaint(img_b64_raw, mask_b64_raw, prompt, seed)
    return call_comfy(img_b64_raw, mask_b64_raw, prompt, seed)

def _generate_variations(img: Image.Image, style: str, mode: str, n: int = 5) -> List[Dict[str, Any]]:
    prompt = build_prompt(style)

    # 1) High-quality mask from original PNG
    mask_b64, _ = call_mask_worker(
        mode=mode,
        img_b64=b64_png(img),
        feather_px=1.2 if mode == "preview" else 1.5,
        decontaminate=(mode == "final")
    )

    # 2) Render input image — clamp and JPEG to keep payload small
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_render = b64_jpeg(img_small, q=90)

    # 3) N seeds → images
    seeds = [secrets.randbits(32) for _ in range(n)]
    results: List[Dict[str, Any]] = []
    for s in seeds:
        imgs = _render_once(img_b64_for_render, mask_b64, prompt, seed=s)
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})
    return results

# -------- endpoints for previews/zip --------
@app.post("/preview/json")
async def preview_json(file: UploadFile = File(...), style: str = Form(DEFAULT_STYLE_KEY), mode: str = Form("preview")):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    results = _generate_variations(img, style, mode, n=5)
    return {"mode": mode, "style": style, "prompt_used": build_prompt(style), "count": len(results), "results": results}

@app.post("/preview/html", response_class=HTMLResponse)
async def preview_html(file: UploadFile = File(...), style: str = Form(DEFAULT_STYLE_KEY), mode: str = Form("preview")):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    results = _generate_variations(img, style, mode, n=5)
    html = "<h1>Generated Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for r in results:
        html += f"<figure style='margin:8px'><img src='data:image/png;base64,{r['image_b64']}' style='width:300px;margin:5px;display:block'/></figure>"
    html += "</div>"
    return HTMLResponse(content=html)

@app.post("/preview/zip")
async def preview_zip(file: UploadFile = File(...), style: str = Form(DEFAULT_STYLE_KEY), mode: str = Form("preview")):
    """Generate as /preview/json then bundle to a local ZIP for download."""
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    results = _generate_variations(img, style, mode, n=5)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(results, 1):
            zf.writestr(f"mockup_{i:02d}.png", base64.b64decode(r["image_b64"]))
    mem.seek(0)
    headers = {"Content-Disposition": "attachment; filename=mockups.zip"}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)
