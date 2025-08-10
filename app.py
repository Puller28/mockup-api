import os
import io
import base64
import json
from typing import List, Dict, Tuple

from PIL import Image, ImageOps, ImageDraw

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
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
# Canvas is 1024x1024. Place the framed art at (x,y,w,h).
# Tune the rects to taste per template.
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
    "blurry, low detail, distorted, bad framing, artifacts, low quality, overexposed, underexposed"
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
app = FastAPI(title="Mockup Generator (Outpaint Around Art)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class BatchResponse(BaseModel):
    template: str
    prompt: str
    images: List[str]  # data URLs (PNG)

# =========================
# Image utils
# =========================
CANVAS_SIZE = (1024, 1024)

def _resize_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize img to fit inside (target_w,target_h) keeping aspect (letterbox if needed)."""
    return ImageOps.contain(img, (target_w, target_h), method=Image.LANCZOS)

def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def _to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

def _compose_art_and_mask(upload_bytes: bytes, rect: Tuple[int, int, int, int]) -> Tuple[bytes, bytes]:
    """
    Build:
      - art_canvas.png  (art pasted into the fixed frame rectangle on a neutral 1024x1024)
      - mask.png        (black = keep art; white = generate background)
    Returns the PNG bytes for both.
    """
    x, y, w, h = rect

    # Load upload
    art = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
    art_resized = _resize_fit(art, w, h)

    # Create blank canvas (neutral gray)
    canvas = Image.new("RGB", CANVAS_SIZE, (128, 128, 128))
    # Center letterbox within rect if art_resized doesn't fill w×h exactly
    paste_x = x + (w - art_resized.width) // 2
    paste_y = y + (h - art_resized.height) // 2
    canvas.paste(art_resized, (paste_x, paste_y))

    # Make mask: black over the EXACT rect area where art could live, white elsewhere
    # (We preserve the entire rect region strictly; background is generated outside it.)
    mask = Image.new("L", CANVAS_SIZE, 255)  # white = inpaint/generate
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + w, y + h], fill=0)  # black = keep

    return _png_bytes(canvas), _png_bytes(mask.convert("RGB"))

# =========================
# RunPod calls
# =========================
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
    """Normalize image outputs (prefer base64 or urls). Returns data URLs if base64 is present."""
    out = (status_payload or {}).get("output") or {}
    results: List[str] = []

    imgs = out.get("images")
    if isinstance(imgs, list):
        for it in imgs:
            if isinstance(it, dict):
                if it.get("base64"):
                    results.append("data:image/png;base64," + it["base64"])
                elif it.get("content"):
                    results.append("data:image/png;base64," + it["content"])
                elif it.get("url"):
                    results.append(it["url"])
                elif it.get("path"):
                    results.append(it["path"])
            elif isinstance(it, str):
                if it.startswith("http"):
                    results.append(it)
                else:
                    results.append("data:image/png;base64," + it)
    if results:
        return results

    urls = out.get("urls")
    if isinstance(urls, list) and urls:
        return urls

    b64s = out.get("base64")
    if isinstance(b64s, list) and b64s:
        return ["data:image/png;base64," + b for b in b64s if isinstance(b, str) and b]

    if isinstance(out.get("image_url"), str) and out["image_url"]:
        return [out["image_url"]]

    if isinstance(out.get("image_path"), str) and out["image_path"]:
        return [out["image_path"]]

    return []

def build_inpaint_workflow(prompt: str, seed: int) -> dict:
    """
    ComfyUI graph (no transformation of artwork):
      LoadImage('art.png') + LoadImageMask('mask.png')
      -> VAEEncodeForInpaint (grow_mask_by=24)
      -> KSampler (denoise=1.0)
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
            "img": {  # load composed canvas with art
                "class_type": "LoadImage",
                "inputs": {"image": "art.png"}
            },
            "msk": {  # load mask (black keep, white generate)
                "class_type": "LoadImageMask",
                "inputs": {"image": "mask.png"}
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
                    "steps": 28,
                    "cfg": 6.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "dec": {  # decode
                "class_type": "VAEDecode",
                "inputs": {"samples": ["ks", 0], "vae": ["100", 2]}
            },
            "save": {  # save final
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
    return {"message": "Mockup API ready", "templates": list(TEMPLATES.keys())}

@app.post("/batch", response_model=BatchResponse)
async def batch(template: str = Form(...), file: UploadFile = File(...)):
    """Return 5 mockups as data URLs (PNG)."""
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Invalid template. Options: {list(TEMPLATES.keys())}")

    rect = tuple(TEMPLATES[template]["rect"])
    prompt_text = TEMPLATES[template]["prompt"]

    # read upload
    raw = await file.read()

    # compose canvas + mask
    art_png, mask_png = _compose_art_and_mask(raw, rect)
    art_b64 = base64.b64encode(art_png).decode("utf-8")
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    images_all: List[str] = []
    for i in range(5):
        # build workflow for this seed
        wf = build_inpaint_workflow(prompt_text, seed=123456 + i)

        # RunPod "runsync" expects images list with {name,image}
        payload = {
            "input": {
                "return_type": "base64",
                **wf,
                "images": [
                    {"name": "art.png", "image": art_b64},
                    {"name": "mask.png", "image": mask_b64},
                ],
            }
        }

        result = call_runsync(payload, timeout_sec=420)

        # log one raw sample for troubleshooting
        if i == 0:
            try:
                print("RUNPOD_RAW_SAMPLE:", json.dumps(result)[:4000])
            except Exception:
                print("RUNPOD_RAW_SAMPLE: <non-serializable>")

        outs = extract_images_from_output(result)
        images_all.append(outs[0] if outs else "MISSING")

    return BatchResponse(template=template, prompt=prompt_text, images=images_all)

@app.post("/batch/html")
async def batch_html(template: str = Form(...), file: UploadFile = File(...)):
    """Quick visual check in the browser (POST with multipart form from Swagger)."""
    data = await batch(template, file)
    html = [f"<h2>{template}</h2><p>{TEMPLATES[template]['prompt']}</p>"]
    for idx, src in enumerate(data.images, 1):
        html.append(f"<div style='margin:10px 0'><strong>{idx}</strong><br><img style='max-width:640px' src='{src}'/></div>")
    return Response("\n".join(html), media_type="text/html")
