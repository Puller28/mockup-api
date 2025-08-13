# app.py — stable rollback (mask worker + comfy /run + local ZIP)

import io, os, json, time, base64, secrets, zipfile
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image

app = FastAPI(
    title="Mockup API (Stable Rollback)",
    description="Mask worker + Comfy /run, with local ZIP bundling.",
    version="2.4"
)

# ---------------- Env ----------------
RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT    = os.getenv("RUNPOD_MASK_ENDPOINT")     # .../<MASK_ID>/runsync
COMFY_ENDPOINT   = os.getenv("RUNPOD_COMFY_ENDPOINT")    # .../<COMFY_ID>/run
COMFY_MODEL      = os.getenv("RUNPOD_COMFY_MODEL", "v1-5-pruned-emaonly.safetensors")
WARMUP_ENABLED   = os.getenv("WARMUP_ENABLED", "false").lower() == "true"

# --- Exec (serverless) optional path ---
EXEC_ENDPOINT   = os.getenv("RUNPOD_EXEC_ENDPOINT")         # e.g. .../<EXEC_ID>/runsync or /run
USE_EXEC        = os.getenv("USE_EXEC", "false").lower() == "true"


def _assert_env():
    missing = []
    if not RUNPOD_API_KEY: missing.append("RUNPOD_API_KEY")
    if not MASK_ENDPOINT:  missing.append("RUNPOD_MASK_ENDPOINT")

    # Option A (exec) takes precedence when USE_EXEC = true
    if USE_EXEC:
        if not EXEC_ENDPOINT:
            missing.append("RUNPOD_EXEC_ENDPOINT")
    else:
        if not COMFY_ENDPOINT:
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
    """Defensive: ensure a clean RGBA PNG alpha mask."""
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
    
def _strip_data_url(b64_or_data_url: str) -> str:
    """If a 'data:image/...;base64,xxxx' string slips in, strip the header."""
    if isinstance(b64_or_data_url, str) and b64_or_data_url.startswith("data:"):
        try:
            return b64_or_data_url.split(",", 1)[1]
        except Exception:
            return b64_or_data_url
    return b64_or_data_url

# ---------- Debug helpers for mask visualization ----------

def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _img_to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(img)).decode("utf-8")

def _to_rgba(im: Image.Image) -> Image.Image:
    return im.convert("RGBA")

def _make_checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    bg = Image.new("RGB", (w, h), (200, 200, 200))
    px = bg.load()
    for y in range(h):
        for x in range(w):
            if ((x // tile) + (y // tile)) % 2 == 0:
                px[x, y] = (220, 220, 220)
            else:
                px[x, y] = (180, 180, 180)
    return bg

def _apply_mask_cutout(src_rgba: Image.Image, mask_rgba: Image.Image, invert: bool = False) -> tuple[Image.Image, Image.Image]:
    """
    Returns (subject_cutout_rgba, background_only_rgba)
    - subject_cutout: transparent where mask is opaque (background), original where mask is transparent (subject kept)
    - background_only: original where mask is opaque (background), transparent where mask is transparent (subject)
    """
    w, h = src_rgba.size
    # ensure mask alpha channel
    if "A" in mask_rgba.getbands():
        alpha = mask_rgba.getchannel("A")
    else:
        alpha = mask_rgba.convert("L")  # fallback luminance

    if invert:
        alpha = Image.eval(alpha, lambda a: 255 - a)

    # background_only: keep where alpha>0
    bg_only = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg_only = Image.composite(src_rgba, bg_only, alpha)

    # subject_cutout: keep inverse of alpha
    inv_alpha = Image.eval(alpha, lambda a: 255 - a)
    subj = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    subj = Image.composite(src_rgba, subj, inv_alpha)

    return subj, bg_only


# ------------- HTTP helpers ---------------
def _headers_json() -> Dict[str, str]:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def call_runpod(endpoint: str, payload: dict | str) -> dict:
    """Safely POST JSON without double-encoding; tolerant if payload is a string."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            try:
                r = requests.post(endpoint, headers=_headers_json(), data=payload, timeout=180)
            except requests.RequestException as e:
                raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint}: {e}")
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
            return r.json()

    try:
        r = requests.post(endpoint, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}, json=payload, timeout=180)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint}: {r.text}")
    return r.json()

# ---------- Mask worker ----------
def call_mask_worker(img_b64: str, mode: str = "preview") -> str:
    payload = {"input": {"image_b64": img_b64, "mode": mode}}
    data = call_runpod(MASK_ENDPOINT, payload)
    out = data.get("output") or {}
    mask_b64 = out.get("mask_b64") or data.get("mask_b64")
    if not mask_b64:
        raise HTTPException(status_code=502, detail=f"Mask worker returned no mask: {data}")
    return normalize_mask_b64(mask_b64)

# ---------- Exec worker (renderer) ----------
def call_exec(img_b64: str, mask_b64: str, prompt: str, seed: int | None = None) -> list[str]:
    """
    Calls your RunPod EXEC worker (serverless handler that accepts base64).
    Supports /runsync or /run depending on EXEC_ENDPOINT.
    """
    if not EXEC_ENDPOINT:
        raise HTTPException(status_code=500, detail="RUNPOD_EXEC_ENDPOINT not set")

    payload = {
        "input": {
            "image_b64": img_b64,
            "mask_b64":  mask_b64,
            "prompt":    prompt
        }
    }
    if seed is not None:
        payload["input"]["seed"] = int(seed)

    ep = EXEC_ENDPOINT.rstrip("/")
    if ep.endswith("/runsync"):
        data = call_runpod(ep, payload)
    elif ep.endswith("/run"):
        data = runpod_run_and_wait(ep, payload)
    else:
        # default to runsync if suffix unknown
        data = call_runpod(ep, payload)

    out = data.get("output") or {}
    # common returns: {"output":{"image_b64": "..."}}
    if isinstance(out, dict) and "image_b64" in out:
        return [out["image_b64"]]
    # or {"output":{"images":[{"image":"..."}, ...]}}
    if isinstance(out, dict) and "images" in out:
        return [x["image"] if isinstance(x, dict) and "image" in x else x for x in out["images"]]
    # or {"output": ["...", "..."]}
    if isinstance(out, list):
        return out

    raise HTTPException(status_code=502, detail=f"Exec worker returned no images: {data}")


    data = call_runpod(COMFY_ENDPOINT, payload)  # COMFY_ENDPOINT now points to EXEC /runsync

    # Common shapes:
    # 1) {"output": {"image_b64": "..."}}
    # 2) {"output": {"images": [{"image": "..."}, ...]}}
    # 3) {"output": ["...", "..."]}
    out = data.get("output") or {}
    if isinstance(out, dict) and "image_b64" in out:
        return [out["image_b64"]]
    if isinstance(out, dict) and "images" in out:
        return [x["image"] if isinstance(x, dict) and "image" in x else x for x in out["images"]]
    if isinstance(out, list):
        return out

    raise HTTPException(status_code=502, detail=f"Exec worker returned no images: {data}")


    if EXEC_ENDPOINT.rstrip("/").endswith("/runsync"):
        data = call_runpod(EXEC_ENDPOINT, payload)
    else:
        data = runpod_run_and_wait(EXEC_ENDPOINT, payload)

    out = data.get("output") or {}
    images = out.get("images", [])
    if not images and isinstance(out, list):               # defensive fallback
        images = [{"image": x} for x in out]
    if not images:
        raise HTTPException(status_code=502, detail=f"Exec worker returned no images: {data}")
    return [e["image"] for e in images if isinstance(e, dict) and "image" in e]


# ---------- ComfyUI workflow ----------
def comfy_workflow(seed: int, prompt: str) -> Dict[str, Any]:
    """
    Vanilla inpaint pipeline (works on stock Comfy):
      CheckpointLoaderSimple -> model/clip/vae
      LoadImage + LoadImageMask (base64 objects)
      VAEEncodeForInpaint -> KSampler (mask-aware) -> VAEDecode -> SaveImage
    """
    return {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": COMFY_MODEL}
        },

        # IMPORTANT: provide base64 as an object {image: "...", type: "base64"}
        "img": {
            "class_type": "LoadImage",
            "inputs": {
                "image": {"image": "__b64_img__", "type": "base64"}
            }
        },

        "mask": {
            "class_type": "LoadImageMask",
            "inputs": {
                "image": {"image": "__b64_mask__", "type": "base64"},
                "channel": "alpha"
            }
        },

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

def runpod_run_and_wait(endpoint_run: str, payload: dict, timeout_s: int = 240, poll_every: float = 1.5) -> dict:
    """POST to /run, then poll /status/{id} until COMPLETED/FAILED."""
    # submit
    try:
        r = requests.post(endpoint_run, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RunPod request failed to {endpoint_run}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error from {endpoint_run}: {r.text}")

    job = r.json()
    job_id = job.get("id") or job.get("output", {}).get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod /run did not return a job id: {job}")

    # poll
    base = endpoint_run.rstrip("/").rsplit("/", 1)[0]  # strip "/run"
    status_url = f"{base}/status/{job_id}"
    started = time.time()

    while True:
        try:
            s = requests.get(status_url, headers=_headers_json(), timeout=30)
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


def call_comfy(img_b64_for_comfy: str, mask_b64: str, prompt: str, seed: int) -> List[str]:
    # img_b64_for_comfy and mask_b64 are expected to be **data URLs**
    wf = json.loads(
        json.dumps(comfy_workflow(seed, prompt))
        .replace("__b64_img__", img_b64_for_comfy)
        .replace("__b64_mask__", mask_b64)
    )
    payload = {"input": {"return_type": "base64", "workflow": wf}}

    # IMPORTANT: /run is async — wait for completion
    data = runpod_run_and_wait(COMFY_ENDPOINT, payload)

    out = data.get("output") or {}
    images = out.get("images", [])
    if not images and isinstance(out, list):
        images = [{"image": x} for x in out]
    if not images:
        raise HTTPException(status_code=502, detail=f"Comfy returned no images: {data}")
    return [e["image"] for e in images if isinstance(e, dict) and "image" in e]

def call_exec_inpaint(img_b64: str, mask_b64: str, prompt: str, seed: int) -> list[str]:
    """
    Call the exec worker that takes base64 directly and returns base64 images.
    Works with either /runsync or /run endpoints.
    """
    if not EXEC_ENDPOINT:
        raise HTTPException(status_code=500, detail="RUNPOD_EXEC_ENDPOINT not set")

    payload = {
        "input": {
            "image_b64": img_b64,
            "mask_b64":  mask_b64,
            "prompt":    prompt,
            "seed":      seed,
            "return_type": "base64"
        }
    }

    ep = EXEC_ENDPOINT.rstrip("/")
    if ep.endswith("/runsync"):
        data = call_runpod(ep, payload)
    elif ep.endswith("/run"):
        data = runpod_run_and_wait(ep, payload)
    else:
        # assume runsync by default
        data = call_runpod(ep, payload)

    out = data.get("output") or {}
    images = out.get("images", [])
    # Some execs return straight list
    if isinstance(out, list):
        images = out
    if not images:
        raise HTTPException(status_code=502, detail=f"Exec worker returned no images: {data}")
    # items can be dicts or raw strings; normalize
    result = []
    for e in images:
        if isinstance(e, dict) and "image" in e:
            result.append(e["image"])
        elif isinstance(e, str):
            result.append(e)
    return result


# -------------- Routes --------------
@app.get("/debug/env")
def debug_env():
    return {
        "has_api_key": bool(RUNPOD_API_KEY),
        "mask_endpoint": MASK_ENDPOINT,
        "comfy_endpoint": COMFY_ENDPOINT,
        "exec_endpoint": EXEC_ENDPOINT,
        "use_exec": USE_EXEC,
        "active_render_endpoint": EXEC_ENDPOINT if USE_EXEC else COMFY_ENDPOINT,
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
    mask_b64 = call_mask_worker(b64_png(img), mode=mode)
    return {"mode": mode, "mask_b64": mask_b64}

def _generate_variations(img: Image.Image, style: str, mode: str, n: int = 5) -> List[Dict[str, Any]]:
    prompt = build_prompt(style)

    # 1) Mask: best edges when we send PNG of the original
    mask_b64 = call_mask_worker(b64_png(img), mode=mode)

    # 2) Exec worker inference image: clamp + JPEG to keep payload small
    img_small = clamp_image(img, max_side=1280)
    img_b64_for_infer = b64_jpeg(img_small, q=90)

    # 3) Generate N variations by seeding
    seeds = [secrets.randbits(32) for _ in range(n)]
    results: List[Dict[str, Any]] = []
    for s in seeds:
        imgs = call_exec(img_b64_for_infer, mask_b64, prompt, seed=s)  # EXEC worker
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})
    return results

from fastapi.responses import Response

# keep last artifacts around for quick download
_last_mask_png: bytes | None = None
_last_cutout_png: bytes | None = None

@app.post("/debug/mask", response_class=HTMLResponse)
async def debug_mask(
    file: UploadFile = File(...),
    mode: str = Form("preview"),
    invert: int = Form(0)
):
    """
    Show the raw mask from the mask worker and how it applies to the image.
    - mode: preview|final (forwarded to mask worker)
    - invert: 0 or 1 (visually flip mask to test inversion issues)
    """
    _assert_env()

    # 1) load original image
    src = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    w, h = src.size

    # 2) get mask from mask-worker
    mask_b64 = call_mask_worker(b64_png(src), mode=mode)
    # decode to image (should be RGBA with alpha)
    mask_bytes = base64.b64decode(mask_b64)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")

    # 3) optional invert for debugging
    inv = bool(invert)
    cutout, bg_only = _apply_mask_cutout(src, mask, invert=inv)

    # 4) show mask itself on checkerboard to see alpha properly
    checker = _make_checkerboard(w, h)
    checker_rgba = checker.convert("RGBA")
    # mask’s alpha used to composite a white layer for visibility
    white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    mask_over_checker = Image.composite(white, checker_rgba, mask.getchannel("A"))

    # 5) keep last outputs for easy download
    global _last_mask_png, _last_cutout_png
    _last_mask_png   = _png_bytes(mask)
    _last_cutout_png = _png_bytes(cutout)

    # 6) build HTML
    html = f"""
    <h1>Mask Debug</h1>
    <p><b>mode:</b> {mode} &nbsp;&nbsp; <b>invert:</b> {inv}</p>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <figure style="margin:0">
        <figcaption>Original</figcaption>
        <img style="max-width:380px" src="{_img_to_data_url(src)}"/>
      </figure>
      <figure style="margin:0">
        <figcaption>Raw Mask (PNG)</figcaption>
        <img style="max-width:380px" src="{_img_to_data_url(mask)}"/>
      </figure>
      <figure style="margin:0">
        <figcaption>Mask on Checkerboard (alpha)</figcaption>
        <img style="max-width:380px" src="{_img_to_data_url(mask_over_checker)}"/>
      </figure>
      <figure style="margin:0">
        <figcaption>Subject Cutout (kept area)</figcaption>
        <img style="max-width:380px" src="{_img_to_data_url(cutout)}"/>
      </figure>
      <figure style="margin:0">
        <figcaption>Background-Only (to be inpainted)</figcaption>
        <img style="max-width:380px" src="{_img_to_data_url(bg_only)}"/>
      </figure>
    </div>
    <p>
      <a href="/debug/last-mask.png" target="_blank">Download last mask</a> |
      <a href="/debug/last-cutout.png" target="_blank">Download last cutout</a>
    </p>
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
        call_runpod(MASK_ENDPOINT, {"input": {"image_b64": _TINY_PNG_B64, "mode": "preview"}})
    except Exception:
        pass
    try:
        if USE_EXEC and EXEC_ENDPOINT:
            payload = {"input": {"image_b64": _TINY_PNG_B64, "mask_b64": _TINY_PNG_B64, "prompt": "warmup", "seed": 1, "return_type": "base64"}}
            _ = call_runpod(EXEC_ENDPOINT, payload) if EXEC_ENDPOINT.endswith("/runsync") else runpod_run_and_wait(EXEC_ENDPOINT, payload)
    except Exception:
        pass
    return {"ok": True, "warmed": True}


@app.post("/preview/json")
async def preview_json(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY),
    mode: str = Form("preview")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    results = _generate_variations(img, style, mode, n=5)
    return {"mode": mode, "style": style, "prompt_used": build_prompt(style), "count": len(results), "results": results}

@app.post("/preview/html", response_class=HTMLResponse)
async def preview_html(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY),
    mode: str = Form("preview")
):
    _assert_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    results = _generate_variations(img, style, mode, n=5)
    html = "<h1>Generated Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for r in results:
        html += f"<figure style='margin:8px'><img src='data:image/png;base64,{r['image_b64']}' style='width:300px;margin:5px;display:block'/></figure>"
    html += "</div>"
    return HTMLResponse(content=html)

@app.post("/preview/zip")
async def preview_zip(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE_KEY),
    mode: str = Form("preview")
):
    """Local ZIP: generate images exactly like /preview/json, then bundle them."""
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


