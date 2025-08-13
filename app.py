# app.py — Outpainted Wall-Art Mockups (single file)
# Upload art -> we protect it with a mask -> GPT Image API outpaints a room+frame around it.

import os, io, base64
from typing import Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from openai import OpenAI

# ---------------- Config ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Generation resolution; 1536–3072 are reasonable. 2048 is a good default.
DEFAULT_TARGET_PX = int(os.getenv("TARGET_PX", "2048"))

# Optional size presets (approx). We return previews for these if requested.
PRINT_SIZES: Dict[str, Tuple[int, int]] = {
    "4x5":   (1600, 2000),
    "3x4":   (1536, 2048),
    "2x3":   (1600, 2400),
    "11x14": (1650, 2100),
    "A4":    (1654, 2339),  # ~300 DPI
}

# Style prompts — keep them tight and literal to avoid clutter.
STYLE_PROMPTS: Dict[str, str] = {
    "modern_living_black_frame_spotlit": (
        "Create a realistic interior mockup AROUND the existing artwork. "
        "Keep the artwork exactly as-is. Paint a modern living room scene: "
        "matte neutral wall with subtle texture, evening ambience, a focused ceiling spotlight, "
        "a thin black metal frame around the artwork with correct perspective and natural shadows. "
        "No text, no extra logos, tasteful minimalist styling."
    ),
    "gallery_white": (
        "Create a clean gallery wall around the existing artwork. White wall with subtle texture, "
        "museum lighting from above, thin black frame, soft shadows, no text or labels."
    ),
    "bedroom_soft_wood": (
        "Around the existing artwork, paint a cozy bedroom scene: warm neutral wall, soft daylight, "
        "light wood frame, gentle shadows. Keep the artwork unchanged."
    ),
}

# ---------------- App ----------------
app = FastAPI(title="Outpainted Wall-Art Mockups", version="1.0")

# (Optional) relax CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True, "styles": list(STYLE_PROMPTS.keys())}

# ---------------- Helpers ----------------
def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def _resize_fit(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    # Preserve aspect ratio; fit INSIDE the target box.
    return ImageOps.contain(img, target, Image.LANCZOS)

def _pad_canvas_keep_center(img: Image.Image,
                            pad_ratio: float,
                            target_side: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    - Optionally upscales the art to target_side (longest edge)
    - Adds a border pad around it (pad_ratio of max dimension)
    - Returns (padded_canvas_RGBA, bbox_of_art_on_canvas)
    """
    img = img.convert("RGBA")

    # Upscale small originals a bit so the model has pixels to work with
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
    For the Images Edits API:
    - Transparent = KEEP original (the art)
    - Opaque      = PAINT new content (the border/room)
    """
    W, H = canvas_size
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 255))  # paint everywhere
    x0, y0, x1, y1 = keep_bbox
    hole = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))  # transparent hole
    mask.paste(hole, (x0, y0))
    return mask

# ---------------- Route ----------------
@app.post("/outpaint/mockup")
async def outpaint_mockup(
    file: UploadFile = File(...),
    style_key: str = Form("modern_living_black_frame_spotlit"),
    target_px: int = Form(DEFAULT_TARGET_PX),     # output canvas (square) long side
    pad_ratio: float = Form(0.42),                # how much room around the art (0.35–0.5 typical)
    make_print_previews: int = Form(1)            # 1 = include PRINT_SIZES previews in response
):
    """
    Upload a piece of art; we generate a realistic room + frame around it.
    Returns a PNG (base64) and optional previews for common print sizes.
    """
    if style_key not in STYLE_PROMPTS:
        raise HTTPException(400, f"Unknown style_key. Choose one of: {list(STYLE_PROMPTS)}")

    try:
        art = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    # 1) Prepare padded canvas and mask (protect original art)
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    canvas_bytes = _img_to_png_bytes(canvas)
    mask_bytes = _img_to_png_bytes(mask)

    # 2) Call Images API (edits / outpainting)
    prompt = STYLE_PROMPTS[style_key]
    try:
        resp = client.images.edits(
            model="gpt-image-1",
            image=canvas_bytes,
            mask=mask_bytes,
            prompt=prompt,
            size=f"{canvas.width}x{canvas.height}",
            n=1
        )
    except Exception as e:
        raise HTTPException(502, f"Image API error: {e}")

    b64 = resp.data[0].b64_json
    outpainted = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

    # 3) Optional print-size previews
    previews: Dict[str, str] = {}
    if make_print_previews:
        for name, wh in PRINT_SIZES.items():
            img = _resize_fit(outpainted, wh)
            buf = io.BytesIO(); img.save(buf, "PNG")
            previews[name] = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 4) Return JSON (master + previews)
    master_buf = io.BytesIO()
    outpainted.save(master_buf, "PNG")
    master_b64 = base64.b64encode(master_buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "style": style_key,
        "canvas_size": [canvas.width, canvas.height],
        "art_bbox": keep_bbox,   # (x0, y0, x1, y1) of the protected art region
        "image_b64": master_b64,
        "previews": previews
    })
