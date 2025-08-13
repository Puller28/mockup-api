# app.py — Outpainted Wall-Art Mockups (REST, ratio+mat, PNG/ZIP, size=auto)
# Upload art -> protect art with mask -> OpenAI Images/edits paints room+frame around it.

import io
import os
import base64
import requests
import zipfile
from math import ceil
from typing import Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

app = FastAPI(title="Outpainted Wall-Art Mockups (REST)", version="1.3")

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

def _pad_to_ratio(img: Image.Image, ratio_w: int, ratio_h: int, bg=(0, 0, 0, 0)):
    """
    Pads image (no crop) to the given aspect ratio by adding transparent borders.
    Returns (padded_img, placed_bbox) where placed_bbox=(x0,y0,x1,y1) of original on new canvas.
    """
    img = img.convert("RGBA")
    w, h = img.size
    target_ratio = ratio_w / ratio_h
    src_ratio = w / h

    if abs(src_ratio - target_ratio) < 1e-6:
        canvas = Image.new("RGBA", (w, h), bg)
        canvas.paste(img, (0, 0), img)
        return canvas, (0, 0, w, h)

    if src_ratio > target_ratio:
        # wider than target: extend height
        canvas_w = w
        canvas_h = ceil(w / target_ratio)
    else:
        # taller than target: extend width
        canvas_h = h
        canvas_w = ceil(h * target_ratio)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg)
    x0 = (canvas_w - w) // 2
    y0 = (canvas_h - h) // 2
    canvas.paste(img, (x0, y0), img)
    return canvas, (x0, y0, x0 + w, y0 + h)

def _pad_canvas_keep_center(img: Image.Image,
                            pad_ratio: float,
                            target_side: int):
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
    Correct for Images Edits API:
      - Transparent = PAINT (model edits here)  -> the border around the art
      - Opaque      = KEEP  (preserve content)  -> the original artwork region
    """
    W, H = canvas_size
    x0, y0, x1, y1 = keep_bbox

    # Start fully transparent so the model can PAINT the room everywhere...
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # ...then paste an OPAQUE block where the artwork is, so it is KEPT.
    keep_block = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 255))
    mask.paste(keep_block, (x0, y0))

    return mask


# ---- OpenAI Images/edits via REST (version-proof) ----
def images_edit_rest(image_bytes: bytes, mask_bytes: bytes, prompt: str, size: str = "auto") -> str:
    """
    Calls https://api.openai.com/v1/images/edits (multipart).
    Text fields go in `data`, binaries in `files`. Returns first image's base64.
    Uses size='auto' by default so any aspect ratio works.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        headers["OpenAI-Organization"] = org_id

    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": size,   # 'auto' | '1024x1024' | '1024x1536' | '1536x1024'
        # "n": "1",
    }
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
    make_print_previews: int = Form(1),      # 1=yes, 0=no
    normalize_ratio: str = Form(""),         # e.g. "4:5", "3:4", "2:3" ("" = keep original aspect)
    mat_pct: float = Form(0.0),              # e.g., 0.03 for 3% inner margin
    return_format: str = Form("json"),       # "json" | "png" | "zip"
    filename: str = Form("mockup"),          # base filename for png/zip
):
    """
    Upload a piece of art; API paints a realistic room + thin frame around it.
    Options:
      - normalize_ratio: force art to a given print ratio (pads only, no crop)
      - mat_pct: inner margin inside the frame (0–0.1 typical)
      - return_format: "json" (default), "png" (direct download), or "zip" (master + previews)
    """
    if style_key not in STYLE_PROMPTS:
        raise HTTPException(400, f"Unknown style_key. Choose one of: {list(STYLE_PROMPTS)}")

    # 1) Read art
    try:
        art = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    # 1a) (optional) normalize to a requested print ratio (pad only)
    if normalize_ratio:
        try:
            rw, rh = [int(x) for x in normalize_ratio.split(":")]
            art, _ = _pad_to_ratio(art, rw, rh)
        except Exception:
            raise HTTPException(400, f"Invalid normalize_ratio '{normalize_ratio}'. Use like '4:5', '3:4', '2:3'.")

    # 1b) (optional) inner mat (shrink visible art a bit to avoid frame touch)
    if mat_pct and mat_pct > 0:
        w, h = art.size
        mx = int(w * mat_pct / 2.0)
        my = int(h * mat_pct / 2.0)
        mat_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mat_canvas.paste(art, (mx, my), art)
        art = mat_canvas

    # 2) Build padded canvas for outpainting (room around the art)
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    # 3) Call OpenAI Images/edits (REST) — default size='auto'
    prompt = STYLE_PROMPTS[style_key]
    b64 = images_edit_rest(
        image_bytes=_img_to_png_bytes(canvas),
        mask_bytes=_img_to_png_bytes(mask),
        prompt=prompt,
        size="auto",  # robust for any aspect ratio
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

    # 5) Respond (json/png/zip)
    buf_master = io.BytesIO()
    master.save(buf_master, "PNG")
    png_bytes = buf_master.getvalue()

    # A) direct PNG stream
    if return_format.lower() == "png":
        headers = {"Content-Disposition": f'inline; filename="{filename}.png"'}
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png", headers=headers)

    # B) ZIP bundle: master + optional previews
    if return_format.lower() == "zip":
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{filename}.png", png_bytes)
            if make_print_previews and previews:
                for name, b64s in previews.items():
                    zf.writestr(f"{filename}_preview_{name}.png", base64.b64decode(b64s))
        mem.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="{filename}.zip"'}
        return StreamingResponse(mem, media_type="application/zip", headers=headers)

    # C) default JSON
    master_b64 = base64.b64encode(png_bytes).decode("utf-8")
    return JSONResponse({
        "style": style_key,
        "canvas_size": [canvas.width, canvas.height],
        "art_bbox": keep_bbox,        # (x0, y0, x1, y1) where original art was preserved
        "normalize_ratio": normalize_ratio,
        "mat_pct": mat_pct,
        "api_size": "auto",
        "image_b64": master_b64,
        "previews": previews
    })
