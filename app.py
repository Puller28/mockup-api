import os
import io
import json
import base64
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops
from typing import List, Dict, Tuple

from PIL import Image, ImageOps, ImageDraw, ImageFilter

# rembg for optional subject-edge refinement (CPU ok on Render)
# If rembg is not available, code still runs (falls back to crisp rect mask).
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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# =========================
# ENV
# =========================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")  # ex: https://api.runpod.ai/v2/<endpoint_id>
DEFAULT_CKPT = os.getenv("CKPT_NAME", "flux1-dev-fp8.safetensors")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
    raise RuntimeError("Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT environment variables.")

# =========================
# TEMPLATE DEFINITIONS
# Canvas is 1024x1024. We place the framed art at (x,y,w,h).
# Tweak these rects to your taste per template; they’re consistent and require no user input.
# =========================
TEMPLATES: Dict[str, Dict] = {
    "bedroom": {
        "prompt": (
            "Framed artwork hanging in a cozy bedroom with sunlight filtering through linen curtains, "
            "photorealistic interior, soft natural light, realistic shadows, DSLR photo."
        ),
        "rect": (192, 62, 640, 900),  # x, y, w, h
    },
    "gallery_wall": {
        "prompt": (
            "Framed print on a gallery wall with spot lighting and minimal decor, photorealistic, "
            "clean plaster wall, realistic shadows."
        ),
        "rect": (222, 162, 580, 700),
    },
    "modern_lounge": {
        "prompt": (
            "Framed artwork in a modern minimalist lounge above a sofa, natural window light, "
            "neutral palette, photorealistic, realistic shadows."
        ),
        "rect": (210, 150, 600, 750),
    },
    "rustic_study": {
        "prompt": (
            "Framed artwork in a rustic study with wooden shelves and a warm desk lamp, cozy lighting, "
            "photorealistic, realistic shadows."
        ),
        "rect": (200, 120, 624, 784),
    },
    "kitchen": {
        "prompt": (
            "Framed botanical print in a bright modern kitchen with plants, daylight, "
            "photorealistic, realistic shadows."
        ),
        "rect": (232, 140, 560, 744),
    },
}

NEGATIVE_PROMPT = (
    "blurry, low detail, distorted, bad framing, artifacts, low quality, overexposed, underexposed, "
    "warped perspective, extra objects, text, watermark, logo"
)

# =========================
# HTTP client with retries
# =========================
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

# =========================
# FastAPI
# =========================
app = FastAPI(title="Mockup Generator — Outpaint Around Uploaded Art")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class BatchResponse(BaseModel):
    template: str
    prompt: str
    images: List[str]  # data URLs (PNG) or HTTP URLs

# =========================
# Image utils
# =========================
CANVAS_SIZE = (1024, 1024)

def _resize_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to fit inside (target_w, target_h) preserving aspect ratio."""
    return ImageOps.contain(img, (target_w, target_h), method=Image.LANCZOS)

def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def _b64_no_prefix(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")

def _to_data_url(b64_or_url: str) -> str:
    if b64_or_url.startswith("http://") or b64_or_url.startswith("https://"):
        return b64_or_url
    if b64_or_url.startswith("data:image/"):
        return b64_or_url
    return "data:image/png;base64," + b64_or_url

def _auto_compose_with_mask(upload_bytes: bytes, rect: Tuple[int, int, int, int]) -> Tuple[bytes, bytes]:
    """
    Build two PNGs (bytes):
      - art.png  : 1024x1024 RGB canvas with the uploaded art sized+placed into rect.
      - mask.png : 1024x1024 RGBA where ALPHA channel is the mask for ComfyUI:
                   alpha=0 over the protected artwork region (keep), alpha=255 elsewhere (paint).
    Optionally refines edges using rembg (if available).
    """
    x, y, w, h = rect

    # Load upload
    art_src = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
    art_fitted = _resize_fit(art_src, w, h)

    # Base canvas (neutral mid-gray to avoid bias)
    base = Image.new("RGB", CANVAS_SIZE, (128, 128, 128))
    paste_x = x + (w - art_fitted.width) // 2
    paste_y = y + (h - art_fitted.height) // 2
    base.paste(art_fitted, (paste_x, paste_y))

    # Start mask as a clean rectangle: alpha 0 over entire frame rect; 255 elsewhere
    mask_alpha = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(mask_alpha)
    draw.rectangle([x, y, x + w, y + h], fill=0)

    # OPTIONAL: refine edges with rembg — make the "keep" (alpha=0) follow the subject edges
    # (We run rembg on the uploaded image only, resize its matte to where the art was pasted)
    if REMBG_AVAILABLE:
        try:
            rgba = Image.open(io.BytesIO(remove(upload_bytes))).convert("RGBA")
            matte = rgba.split()[-1]  # subject alpha: 255 on subject, 0 background
            matte = _resize_fit(matte, art_fitted.width, art_fitted.height)  # match displayed size
            # Place matte where art sits on canvas
            matte_canvas = Image.new("L", CANVAS_SIZE, 0)
            matte_canvas.paste(matte, (paste_x, paste_y))
            # Invert: we want alpha=0 over subject/rect, 255 elsewhere
            matte_inverted = ImageOps.invert(matte_canvas)
            # Fuse with rect keep area: ensure the entire rect stays protected (black/alpha=0)
            # (i.e., take the MIN between rect-black(0) and inverted matte)
            mask_alpha = ImageChops.lighter(mask_alpha, matte_inverted)
            # Feather seams slightly
            mask_alpha = mask_alpha.filter(ImageFilter.GaussianBlur(radius=1.0))
        except Exception:
            # If rembg fails for any reason, just keep the rectangular mask
            pass

    # Build RGBA mask image with alpha channel = mask_alpha
    mask_rgba = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    r = g = b = Image.new("L", CANVAS_SIZE, 0)
    mask_rgba = Image.merge("RGBA", (r, g, b, mask_alpha))

    return _png_bytes(base), _png_bytes(mask_rgba)

# =========================
# RunPod / ComfyUI wiring
# =========================
def _session_headers():
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

def call_runsync(payload: dict, timeout_sec: int = 420) -> dict:
    url = f"{RUNPOD_ENDPOINT}/runsync"
    try:
        r = SESSION.post(url, json=payload, headers=_session_headers(), timeout=timeout_sec)
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
            if isinstance(it, dict):
                if it.get("base64"):
                    results.append(_to_data_url(it["base64"]))
                elif it.get("content"):
                    results.append(_to_data_url(it["content"]))
                elif it.get("url"):
                    results.append(it["url"])
                elif it.get("data"):
                    results.append(_to_data_url(it["data"]))
            elif isinstance(it, str):
                results.append(_to_data_url(it))
        if results:
            return results

    urls = out.get("urls")
    if isinstance(urls, list) and urls:
        return urls

    b64s = out.get("base64")
    if isinstance(b64s, list) and b64s:
        return [_to_data_url(b) for b in b64s if isinstance(b, str) and b]

    if isinstance(out.get("image_url"), str) and out["image_url"]:
        return [out["image_url"]]

    if isinstance(out.get("image_path"), str) and out["image_path"]:
        return [out["image_path"]]

    return []

def build_inpaint_workflow(prompt: str, seed: int) -> dict:
    """
    ComfyUI graph (outpaint around uploaded art):
      LoadImage('art.png') + LoadImageMask('mask.png', channel='alpha')
      -> VAEEncodeForInpaint(grow_mask_by=24)
      -> KSampler(denoise=1.0)
      -> VAEDecode -> SaveImage
    """
    return {
        "workflow": {
            "100": {  # checkpoint
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": DEFAULT_CKPT}
            },
            "pos": {  # positive prompt
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["100", 1], "text": prompt}
            },
            "neg": {  # negative prompt
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["100", 1], "text": NEGATIVE_PROMPT}
            },
            "img": {  # the composed canvas with art
                "class_type": "LoadImage",
                "inputs": {"image": "art.png"}
            },
            "msk": {  # the alpha mask (alpha=0 keep, alpha=255 paint)
                "class_type": "LoadImageMask",
                "inputs": {"image": "mask.png", "channel": "alpha"}
            },
            "enc": {  # encode for inpaint
                "class_type": "VAEEncodeForInpaint",
                "inputs": {"pixels": ["img", 0], "mask": ["msk", 0], "vae": ["100", 2], "grow_mask_by": 24}
            },
            "ks": {  # generate background only
                "class_type": "KSampler",
                "inputs": {
                    "model": ["100", 0],
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "latent_image": ["enc", 0],
                    "seed": seed,
                    "steps": 28,
                    "cfg": 6.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "dec": {  # decode
                "class_type": "VAEDecode",
                "inputs": {"samples": ["ks", 0], "vae": ["100", 2]}
            },
            "save": {  # final image
                "class_type": "SaveImage",
                "inputs": {"images": ["dec", 0], "filename_prefix": "mockup_out"}
            }
        }
    }

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"message": "Mockup API — outpainting around uploaded art", "templates": list(TEMPLATES.keys()),
            "rembg_enabled": REMBG_AVAILABLE}

@app.get("/try", response_class=HTMLResponse)
def try_form():
    return """
    <h2>Mockup Outpainting (POST to /batch or /batch/html)</h2>
    <form action="/batch/html" method="post" enctype="multipart/form-data">
      <label>Template:
        <select name="template">
          <option>bedroom</option>
          <option>gallery_wall</option>
          <option>modern_lounge</option>
          <option>rustic_study</option>
          <option>kitchen</option>
        </select>
      </label>
      <br/><br/>
      <input type="file" name="file" accept="image/*" required />
      <br/><br/>
      <button type="submit">Generate (5 variations)</button>
    </form>
    """

class BatchResponseModel(BaseModel):
    template: str
    prompt: str
    images: List[str]

@app.post("/batch", response_model=BatchResponseModel)
async def batch(template: str = Form(...), file: UploadFile = File(...)):
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Invalid template. Options: {list(TEMPLATES.keys())}")

    rect = tuple(TEMPLATES[template]["rect"])
    prompt_text = TEMPLATES[template]["prompt"]

    raw = await file.read()

    # Compose art on canvas + auto-mask
    art_png, mask_png = _auto_compose_with_mask(raw, rect)
    art_b64 = _b64_no_prefix(art_png)
    mask_b64 = _b64_no_prefix(mask_png)

    images_all: List[str] = []
    for i in range(5):
        wf = build_inpaint_workflow(prompt_text, seed=123456 + i)

        payload = {
            "input": {
                "return_type": "base64",
                **wf,
                "images": [
                    {"name": "art.png", "image": art_b64},
                    {"name": "mask.png", "image": mask_b64}
                ]
            }
        }

        result = call_runsync(payload, timeout_sec=480)

        if i == 0:
            try:
                print("RUNPOD_RAW_SAMPLE:", json.dumps(result)[:4000])
            except Exception:
                print("RUNPOD_RAW_SAMPLE: <non-serializable>")

        outs = extract_images_from_output(result)
        images_all.append(outs[0] if outs else "MISSING")

    return BatchResponseModel(template=template, prompt=prompt_text, images=images_all)

@app.post("/batch/html")
async def batch_html(template: str = Form(...), file: UploadFile = File(...)):
    data = await batch(template, file)
    html = [f"<h2>{template}</h2><p>{TEMPLATES[template]['prompt']}</p>"]
    for idx, src in enumerate(data.images, 1):
        html.append(f"<div style='margin:10px 0'><strong>{idx}</strong><br><img style='max-width:640px' src='{src}'/></div>")
    return Response("\n".join(html), media_type="text/html")
