import os
import io
import json
import base64
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops

# Optional: rembg for nicer subject edges (CPU OK on Render)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError


# ========= ENV =========
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")           # e.g. https://api.runpod.ai/v2/<endpoint_id>
RUNPOD_API_KEY  = os.getenv("RUNPOD_API_KEY")
CKPT_NAME       = os.getenv("CKPT_NAME", "flux1-dev-fp8.safetensors")
TARGET_CANVAS   = int(os.getenv("MOCKUP_CANVAS", "768"))  # 768 is VRAM-friendly; you can set 1024

if not RUNPOD_ENDPOINT or not RUNPOD_API_KEY:
    raise RuntimeError("Please set RUNPOD_ENDPOINT and RUNPOD_API_KEY environment variables.")


# ========= FASTAPI =========
app = FastAPI(title="Mockup Generator — Outpaint Around Art")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ========= HTTP client (retries) =========
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


# ========= Image helpers =========
def _resize_square(img: Image.Image, side: int) -> Image.Image:
    return ImageOps.contain(img, (side, side), method=Image.LANCZOS).resize((side, side), Image.LANCZOS)

def _png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()

def _b64_no_prefix(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")

def _data_url_from_b64(s: str) -> str:
    if s.startswith("http://") or s.startswith("https://") or s.startswith("data:image/"):
        return s
    return "data:image/png;base64," + s

def _upscale_to_1024(img: Image.Image) -> Image.Image:
    if img.width == 1024:
        return img
    return img.resize((1024, 1024), Image.LANCZOS)


# ========= Auto canvas + mask =========
def build_canvas_and_mask(upload_bytes: bytes, side: int) -> tuple[bytes, bytes]:
    """
    Returns:
      - art.png  : square RGB canvas with your image (centered/fit)
      - mask.png : RGBA where ALPHA=255 means "AI may paint here", ALPHA=0 = keep
                   We make background paintable and protect the subject+near edges.
    """
    src = Image.open(io.BytesIO(upload_bytes)).convert("RGBA")
    img = _resize_square(src, side)

    # Base canvas: neutral gray (avoid bias)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(img.convert("RGB"), (0, 0))

    # Default: protect entire image (alpha=0), paint nothing (alpha=0)
    # We'll create a background mask (white/alpha=255) and keep subject area alpha=0.
    if REMBG_AVAILABLE:
        # Subject matte (255=subject, 0=bg)
        matte_bytes = remove(_png_bytes(img), only_mask=True)
        matte = Image.open(io.BytesIO(matte_bytes)).convert("L")
        # Invert for background paint mask (white area = background)
        bg_mask = ImageOps.invert(matte)
        # Slight blur → softer seams
        bg_mask = bg_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    else:
        # No rembg: paint only a thin border (so at least some outpaint happens)
        bg_mask = Image.new("L", (side, side), 0)
        draw = ImageDraw.Draw(bg_mask)
        draw.rectangle([8, 8, side - 8, side - 8], outline=255, width=16)

    # Build RGBA where alpha channel = bg_mask (Comfy: white/opaque = paint)
    mask_rgba = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    r = g = b = Image.new("L", (side, side), 0)
    mask_rgba = Image.merge("RGBA", (r, g, b, bg_mask))

    return _png_bytes(canvas), _png_bytes(mask_rgba)


# ========= ComfyUI workflow + call =========
def build_inpaint_workflow(prompt: str, seed: int) -> dict:
    """
    ComfyUI graph:
      LoadImage('art.png') + LoadImageMask('mask.png', channel='alpha')
      -> VAEEncodeForInpaint(grow_mask_by=24)
      -> KSampler(denoise=1.0, steps=18, cfg=5.5)
      -> VAEDecode -> SaveImage
    """
    return {
        "workflow": {
            "100": {  # checkpoint
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": CKPT_NAME}
            },
            "pos": {  # positive prompt
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["100", 1], "text": prompt}
            },
            "neg": {  # negative prompt
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["100", 1], "text": "blurry, artifacts, low quality, watermark, text"}
            },
            "img": {  # your canvas
                "class_type": "LoadImage",
                "inputs": {"image": "art.png"}  # referenced by name via top-level `images`
            },
            "msk": {  # alpha mask (alpha=paint)
                "class_type": "LoadImageMask",
                "inputs": {"image": "mask.png", "channel": "alpha"}  # needs 'channel'
            },
            "enc": {  # encode for inpaint
                "class_type": "VAEEncodeForInpaint",
                "inputs": {"pixels": ["img", 0], "mask": ["msk", 0], "vae": ["100", 2], "grow_mask_by": 24}
            },
            "ks": {  # background generation only
                "class_type": "KSampler",
                "inputs": {
                    "model": ["100", 0],
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "latent_image": ["enc", 0],
                    "seed": seed,
                    "steps": 18,
                    "cfg": 5.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "dec": {  # decode to image
                "class_type": "VAEDecode",
                "inputs": {"samples": ["ks", 0], "vae": ["100", 2]}
            },
            "save": {  # save final
                "class_type": "SaveImage",
                "inputs": {"images": ["dec", 0], "filename_prefix": "mockup_out"}
            }
        }
    }

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
    out = (status_payload or {}).get("output") or {}
    results: List[str] = []

    imgs = out.get("images")
    if isinstance(imgs, list):
        for it in imgs:
            # common shapes: {"base64": "..."} or {"content":"..."} or {"url":"..."}
            if isinstance(it, dict):
                if it.get("base64"): results.append(_data_url_from_b64(it["base64"]))
                elif it.get("content"): results.append(_data_url_from_b64(it["content"]))
                elif it.get("url"): results.append(it["url"])
                elif it.get("data"): results.append(_data_url_from_b64(it["data"]))
            elif isinstance(it, str):
                results.append(_data_url_from_b64(it))
    if results:
        return results

    urls = out.get("urls")
    if isinstance(urls, list) and urls:
        return urls

    b64s = out.get("base64")
    if isinstance(b64s, list) and b64s:
        return [_data_url_from_b64(b) for b in b64s if isinstance(b, str) and b]

    if isinstance(out.get("image_url"), str) and out["image_url"]:
        return [out["image_url"]]

    if isinstance(out.get("image_path"), str) and out["image_path"]:
        return [out["image_path"]]

    return []


# ========= Core generator (used by both JSON + HTML routes) =========
def generate_variations(upload_bytes: bytes, prompt: str, n: int = 5) -> List[str]:
    # Build canvas + mask (square)
    art_png, mask_png = build_canvas_and_mask(upload_bytes, TARGET_CANVAS)
    art_b64 = _b64_no_prefix(art_png)
    mask_b64 = _b64_no_prefix(mask_png)

    images_all: List[str] = []
    for i in range(n):
        wf = build_inpaint_workflow(prompt, seed=123456 + i)

        payload = {
            "input": {
                "return_type": "base64",
                **wf,
                # IMPORTANT: RunPod ComfyUI wrapper expects uploaded files here
                "images": [
                    {"name": "art.png", "image": art_b64},
                    {"name": "mask.png", "image": mask_b64},
                ]
            }
        }

        result = call_runsync(payload, timeout_sec=480)
        if i == 0:
            try:
                print("RUNPOD_RAW_SAMPLE:", json.dumps(result)[:4000])
            except Exception:
                print("RUNPOD_RAW_SAMPLE:<non-serializable>")

        outs = extract_images_from_output(result)
        if not outs:
            images_all.append("MISSING")
            continue

        # If we generated at 768, upscale for display parity (optional)
        try:
            if outs[0].startswith("data:image/"):
                b64 = outs[0].split(",", 1)[1]
                im = Image.open(io.BytesIO(base64.b64decode(b64)))
                im = _upscale_to_1024(im)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                images_all.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))
            else:
                images_all.append(outs[0])
        except Exception:
            images_all.append(outs[0])

    return images_all


# ========= Routes =========
@app.get("/")
def root():
    return {
        "message": "Mockup API (outpainting around uploaded art)",
        "rembg_enabled": REMBG_AVAILABLE,
        "canvas": TARGET_CANVAS,
        "routes": [r.path for r in app.routes],
    }

@app.get("/try", response_class=HTMLResponse)
def try_form():
    return """
    <h2>Mockup Outpainting — Quick Test</h2>
    <form action="/batch/html" method="post" enctype="multipart/form-data">
      <label>Prompt:</label><br/>
      <input name="prompt" size="80" value="Framed artwork on a cozy bedroom wall, soft window light, realistic shadows, photorealistic interior" />
      <br/><br/>
      <input type="file" name="file" accept="image/*" required />
      <br/><br/>
      <button type="submit">Generate (5 variations)</button>
    </form>
    """

@app.post("/batch")
async def batch(prompt: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    imgs = generate_variations(raw, prompt, n=5)
    return JSONResponse({"status": "ok", "images": imgs})

@app.post("/batch/html")
async def batch_html(prompt: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    imgs = generate_variations(raw, prompt, n=5)
    html = [f"<h3>Prompt</h3><p>{prompt}</p><hr/>"]
    for i, src in enumerate(imgs, 1):
        html.append(f"<div style='margin:10px 0'><b>{i}</b><br/><img style='max-width:640px' src='{src}'/></div>")
    return Response("\n".join(html), media_type="text/html")
