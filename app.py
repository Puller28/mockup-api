# app.py — Outpainted Wall-Art Mockups (REST version, Render-friendly)
# Upload art -> we protect the art with a mask -> OpenAI Images/edits outpaints a room+frame around it.

import io
import os
import base64
import requests
from typing import Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps


# =========================
# Config / Presets
# =========================

DEFAULT_TARGET_PX = int(os.getenv("TARGET_PX", "2048"))  # generation canvas long side
PRINT_SIZES: Dict[str, Tuple[int, int]] = {
    "4x5":   (1600, 2000),
    "3x4":   (1536, 2048),
    "2x3":   (1600, 2400),
    "11x14": (1650, 2100),
    "A4":    (1654, 2339),
}

STYLE_PROMPTS: Dict[str, str] = {
    "modern_living_black_frame_spotlit": (
        "Create a realistic interior mockup AROUND the existing artwork. "
        "Keep the artwork exactly as-is. Paint a modern living room scene: "
        "matte neutral wall with subtle texture, focused ceiling spotlight, "
        "thin black metal frame around the artwork with correct perspective and natural shadows. "
        "Minimalist, photorealistic, no text or logos."
    ),
    "gallery_white": (
        "Create a clean gallery wall around the existing artwork. White wall with subtle texture, "
        "museum lighting from above, thin black frame, soft shadows, no text."
    ),
    "bedroom_soft_wood": (
        "Around the existing artwork, paint a cozy bedroom: warm neutral wall, soft daylight, "
        "light wood frame, gentle shadows. Keep the artwork unchanged, no text."
    ),
}


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Outpainted Wall-Art Mockups (REST)", version="1.1")

# Open CORS for easy testing; tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "styles": list(STYLE_PROMPTS.keys()),
        "target_default": DEFAULT_TARGET_PX,
    }


# =========================
# Helpers
# =========================

def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def _resize_fit(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    return ImageOps.contain(img, target, Image.LANCZOS)

def _pad_canvas_keep_center(img: Image.Image,
                            pad_ratio: float,
                            target_side: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Optionally upscales the art to target_side (longest edge),
    then adds border space around it (pad_ratio of max side).
    Returns (canvas_rgba, art_bbox_on_canvas).
    """
    img = img.convert("RGBA")

    longest = max(img.size)
    scale = max(1.0, target_side / float(longest))
    if scale > 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    w, h = img.size
    border = int(pad_ratio * max(w, h))
    canvas_w, canvas_h = w + 2 * border, h + 2 * border

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x0, y0 = border, border
    canvas.paste(img, (x0, y0), img)

    return canvas, (x0, y0, x0 + w, y0 + h)

def _build_outpaint_mask(canvas_size: Tuple[int, int], keep_bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Mask for Images Edits API:
      - Transparent pixels = KEEP (the original artwork)
      - Opaque pixels      = PAINT (room + frame around)
    """
    W, H = canvas_size
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 255))  # paint everywhere
    x0, y0, x1, y1 = keep_bbox
    hole = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    mask.paste(hole, (x0, y0))
    return mask


# ---- OpenAI Images/edits via REST (version-proof) ----
def images_edit_rest(image_bytes: bytes, mask_bytes: bytes, prompt: str, size: str) -> str:
    """
    Calls https://api.openai.com/v1/images/edits with multipart form.
    Text fields go in `data`, binary images go in `files`.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Text fields must be in `data`
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": size,          # e.g., "2048x2048"
        # "n": "1",            # optional
    }

    # Binary parts go in `files`
    files = {
        "image": ("canvas.png", image_bytes, "image/png"),
        "mask":  ("mask.png",   mask_bytes,  "image/png"),
    }

    try:
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=180)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Images API request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Image API error [{resp.status_code}]: {resp.text}")

    js = resp.json()
    if not js.get("data"):
        raise HTTPException(status_code=502, detail=f"Image API returned no data: {js}")
    return js["data"][0]["b64_json"]


# =========================
# Route: outpaint mockup
# =========================

@app.post("/outpaint/mockup")
async def outpaint_mockup(
    file: UploadFile = File(...),
    style_key: str = Form("modern_living_black_frame_spotlit"),
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    make_print_previews: int = Form(1)  # 1=yes, 0=no
):
    """
    Upload a piece of art; the API paints a realistic room + thin frame around it.
    Returns a master PNG (base64) and optional print-size previews.
    """
    if style_key not in STYLE_PROMPTS:
        raise HTTPException(400, f"Unknown style_key. Choose one of: {list(STYLE_PROMPTS)}")

    # 1) Read art
    try:
        art = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    # 2) Build padded canvas and KEEP mask
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    canvas_bytes = _img_to_png_bytes(canvas)
    mask_bytes = _img_to_png_bytes(mask)

    # 3) Call OpenAI Images/edits (REST)
    prompt = STYLE_PROMPTS[style_key]
    b64 = images_edit_rest(
        image_bytes=canvas_bytes,
        mask_bytes=mask_bytes,
        prompt=prompt,
        size=f"{canvas.width}x{canvas.height}",
    )

    # 4) Decode master + generate optional previews
    master = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

    previews: Dict[str, str] = {}
    if make_print_previews:
        for name, wh in PRINT_SIZES.items():
            img = _resize_fit(master, wh)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            previews[name] = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 5) Respond
    buf_master = io.BytesIO()
    master.save(buf_master, "PNG")
    master_b64 = base64.b64encode(buf_master.getvalue()).decode("utf-8")

    return JSONResponse({
        "style": style_key,
        "canvas_size": [canvas.width, canvas.height],
        "art_bbox": keep_bbox,  # (x0, y0, x1, y1) where the original art was preserved
        "image_b64": master_b64,
        "previews": previews
    })
