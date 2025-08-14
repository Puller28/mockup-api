# app.py — Outpainted Wall-Art Mockups
# - Low-RAM safe ingest (EXIF fix, clamp longest edge, pixel guard)
# - Single-style endpoint (/outpaint/mockup_single) → PNG/JSON
# - Multi-variant endpoint (/outpaint/mockup) → JSON/PNG/ZIP
# - Uses OpenAI Images Edits (gpt-image-1) with size="auto"

import io
import os
import base64
import requests
import zipfile
from math import ceil
from typing import Dict, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFile

# Be tolerant of slightly broken files
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Config / Presets
# =========================

# For 512MB instances these are safe defaults; override via Render env vars.
DEFAULT_TARGET_PX = int(os.getenv("TARGET_PX", "1536"))     # generation canvas long side
MAX_UPLOAD_SIDE   = int(os.getenv("MAX_UPLOAD_SIDE", "1600"))  # clamp uploaded art longest edge
MAX_PIXELS        = int(os.getenv("MAX_PIXELS", "12000000"))   # ~12MP guard (e.g., 4000x3000)

PRINT_SIZES: Dict[str, Tuple[int, int]] = {
    "4x5":   (1600, 2000),
    "3x4":   (1536, 2048),
    "2x3":   (1600, 2400),
    "11x14": (1650, 2100),
    "A4":    (1654, 2339),
}

STYLE_PROMPTS: Dict[str, str] = {
    "living_room": (
        "Create a realistic interior mockup AROUND the existing artwork. Keep the artwork exactly as-is. "
        "Paint a modern living room: matte neutral wall with subtle texture, focused ceiling spotlight, "
        "thin black metal frame with correct perspective and natural shadows. Minimalist, photorealistic, no text."
    ),
    "bedroom": (
        "Around the existing artwork, paint a cozy bedroom: warm neutral wall, soft daylight, "
        "light wood frame around the artwork, gentle shadows. Photorealistic, no text or logos."
    ),
    "study": (
        "Around the existing artwork, paint a refined study / home office: desaturated wall paint, "
        "subtle bookshelf blur and desk hints, muted daylight, thin dark metal frame, soft realistic shadows. No text."
    ),
    "gallery": (
        "Paint a clean gallery wall around the existing artwork: white wall with subtle microtexture, "
        "museum track lighting from above, thin black metal frame, natural falloff shadows. No text."
    ),
    "kitchen": (
        "Around the existing artwork, paint a tasteful kitchen nook: light painted wall, subtle tile / counter hints, "
        "soft daylight, thin black frame, realistic shadows. No text."
    ),
}
DEFAULT_STYLE_LIST = ["living_room", "bedroom", "study", "gallery", "kitchen"]


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Outpainted Wall-Art Mockups (REST)", version="1.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "styles_available": list(STYLE_PROMPTS.keys()),
        "default_styles": DEFAULT_STYLE_LIST,
        "target_default": DEFAULT_TARGET_PX,
        "upload_caps": {"MAX_UPLOAD_SIDE": MAX_UPLOAD_SIDE, "MAX_PIXELS": MAX_PIXELS},
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

def _ingest_user_image(file_bytes: bytes, max_side: int = MAX_UPLOAD_SIDE) -> Image.Image:
    """
    Load the uploaded image safely, apply EXIF orientation,
    clamp longest edge to max_side, enforce pixel-count guard,
    and convert to RGBA for consistent downstream handling.
    """
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)

    # Pixel-count guard: if decompressed size is huge, downscale proportionally
    if img.width * img.height > MAX_PIXELS:
        scale = (MAX_PIXELS / float(img.width * img.height)) ** 0.5
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Clamp longest edge
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def _pad_to_ratio(img: Image.Image, ratio_w: int, ratio_h: int, bg=(0, 0, 0, 0)):
    """Pad (no crop) to ratio_w:ratio_h by adding transparent borders."""
    img = img.convert("RGBA")
    w, h = img.size
    target_ratio = ratio_w / ratio_h
    src_ratio = w / h

    if abs(src_ratio - target_ratio) < 1e-6:
        canvas = Image.new("RGBA", (w, h), bg)
        canvas.paste(img, (0, 0), img)
        return canvas, (0, 0, w, h)

    if src_ratio > target_ratio:
        canvas_w = w
        canvas_h = ceil(w / target_ratio)
    else:
        canvas_h = h
        canvas_w = ceil(h * target_ratio)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg)
    x0 = (canvas_w - w) // 2
    y0 = (canvas_h - h) // 2
    canvas.paste(img, (x0, y0), img)
    return canvas, (x0, y0, x0 + w, y0 + h)

def _pad_canvas_keep_center(img: Image.Image, pad_ratio: float, target_side: int):
    """Upscale to target_side (max edge) then add border by pad_ratio; returns (canvas, art_bbox)."""
    img = img.convert("RGBA")

    # Cap target_side to upload clamp as an extra RAM safeguard
    target_side = min(target_side, MAX_UPLOAD_SIDE)

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

def _build_outpaint_mask(
    canvas_size: Tuple[int, int],
    keep_bbox: Tuple[int, int, int, int],
    lock_pad_px: int = 2,
) -> Image.Image:
    """
    OpenAI Images/edits mask:
      - Transparent (alpha=0) = KEEP ORIGINAL
      - Opaque (alpha=255)    = PAINT / REPLACE

    We make the art area fully transparent, the rest fully opaque.
    """
    W, H = canvas_size
    x0, y0, x1, y1 = keep_bbox

    # Expand keep box a little
    x0 = max(0, x0 - lock_pad_px)
    y0 = max(0, y0 - lock_pad_px)
    x1 = min(W, x1 + lock_pad_px)
    y1 = min(H, y1 + lock_pad_px)

    # Alpha channel: fully opaque everywhere
    alpha = Image.new("L", (W, H), 255)

    # Paste a fully transparent area for the KEEP region
    hole = Image.new("L", (x1 - x0, y1 - y0), 0)
    alpha.paste(hole, (x0, y0))

    # Combine into RGBA (black background, but only alpha matters)
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    mask.putalpha(alpha)
    return mask



def _openai_images_edit(image_bytes: bytes, mask_bytes: bytes, prompt: str, size: str = "auto") -> str:
    """Call OpenAI Images/edits with multipart. Returns first image's base64."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        headers["OpenAI-Organization"] = org_id

    data = {"model": "gpt-image-1", "prompt": prompt, "size": size}
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


# =========================================================
# Low-RAM Endpoint: one style only (recommended)
# =========================================================
@app.post("/outpaint/mockup_single")
async def outpaint_mockup_single(
    file: UploadFile = File(...),
    style: str = Form("living_room"),          # one of STYLE_PROMPTS
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    normalize_ratio: str = Form(""),           # e.g. "4:5", "3:4", "2:3"
    mat_pct: float = Form(0.0),                # 0.0–0.1 typical
    return_format: str = Form("png"),          # "png" | "json"
    filename: str = Form("mockup_single"),
):
    if style not in STYLE_PROMPTS:
        raise HTTPException(400, f"Unknown style '{style}'. Choose from {list(STYLE_PROMPTS.keys())}")

    # Safe ingest (EXIF fix, clamp, pixel guard)
    try:
        art = _ingest_user_image(await file.read(), max_side=MAX_UPLOAD_SIDE)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    if normalize_ratio:
        try:
            rw, rh = [int(x) for x in normalize_ratio.split(":")]
            art, _ = _pad_to_ratio(art, rw, rh)
        except Exception:
            raise HTTPException(400, f"Invalid normalize_ratio '{normalize_ratio}'. Use '4:5', '3:4', '2:3'.")

    if mat_pct and mat_pct > 0:
        w, h = art.size
        mx = int(w * mat_pct / 2.0)
        my = int(h * mat_pct / 2.0)
        mat_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mat_canvas.paste(art, (mx, my), art)
        art = mat_canvas

    # Canvas + mask (with target_px capped internally)
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    b64 = _openai_images_edit(_img_to_png_bytes(canvas), _img_to_png_bytes(mask),
                              prompt=STYLE_PROMPTS[style], size="auto")

    if return_format.lower() == "json":
        return JSONResponse({
            "style": style,
            "canvas_size": [canvas.width, canvas.height],
            "art_bbox": keep_bbox,
            "normalize_ratio": normalize_ratio,
            "mat_pct": mat_pct,
            "api_size": "auto",
            "image_b64": b64
        })

    # default PNG
    img_bytes = base64.b64decode(b64)
    headers = {"Content-Disposition": f'inline; filename="{filename}_{style}.png"'}
    return Response(content=img_bytes, media_type="image/png", headers=headers)


# =========================================================
# Multi-variant Endpoint (use when needed)
#   - return_format=png → renders first style only (fast/low RAM)
#   - ZIP/JSON supported for multiple styles (watch memory)
# =========================================================
@app.post("/outpaint/mockup")
async def outpaint_mockup(
    file: UploadFile = File(...),
    styles: str = Form(",".join(DEFAULT_STYLE_LIST)),   # comma-separated list
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    make_print_previews: int = Form(0),                 # default 0 to save RAM
    normalize_ratio: str = Form(""),
    mat_pct: float = Form(0.0),
    return_format: str = Form("json"),                  # "json" | "png" | "zip"
    filename: str = Form("mockup_bundle"),
):
    style_list = [s.strip() for s in styles.split(",") if s.strip()]
    invalid = [s for s in style_list if s not in STYLE_PROMPTS]
    if invalid:
        raise HTTPException(400, f"Unknown styles {invalid}. Choose from {list(STYLE_PROMPTS.keys())}")
    if not style_list:
        raise HTTPException(400, "No valid styles provided.")

    # Safe ingest
    try:
        art = _ingest_user_image(await file.read(), max_side=MAX_UPLOAD_SIDE)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    if normalize_ratio:
        try:
            rw, rh = [int(x) for x in normalize_ratio.split(":")]
            art, _ = _pad_to_ratio(art, rw, rh)
        except Exception:
            raise HTTPException(400, f"Invalid normalize_ratio '{normalize_ratio}'. Use '4:5', '3:4', '2:3'.")

    if mat_pct and mat_pct > 0:
        w, h = art.size
        mx = int(w * mat_pct / 2.0)
        my = int(h * mat_pct / 2.0)
        mat_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mat_canvas.paste(art, (mx, my), art)
        art = mat_canvas

    # Canvas + mask once
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)
    canvas_bytes = _img_to_png_bytes(canvas)
    mask_bytes   = _img_to_png_bytes(mask)

    # Fast path: PNG → first style only
    if return_format.lower() == "png":
        style = style_list[0]
        b64 = _openai_images_edit(canvas_bytes, mask_bytes, prompt=STYLE_PROMPTS[style], size="auto")
        img_bytes = base64.b64decode(b64)
        headers = {"Content-Disposition": f'inline; filename="{filename}_{style}.png"'}
        return Response(content=img_bytes, media_type="image/png", headers=headers)

    # Multi generation (JSON/ZIP)
    results: List[Dict[str, str]] = []
    for style in style_list:
        try:
            b64 = _openai_images_edit(canvas_bytes, mask_bytes, prompt=STYLE_PROMPTS[style], size="auto")
            results.append({"style": style, "image_b64": b64})
        except HTTPException as e:
            results.append({"style": style, "error": str(e.detail)})

    if not any("image_b64" in r for r in results):
        raise HTTPException(status_code=502, detail={"message": "No mockups generated", "results": results})

    # Optional previews (watch memory)
    previews_map: Dict[str, Dict[str, str]] = {}
    if make_print_previews:
        for item in results:
            if "image_b64" not in item:
                continue
            key = item["style"]
            img = Image.open(io.BytesIO(base64.b64decode(item["image_b64"]))).convert("RGBA")
            pv: Dict[str, str] = {}
            for name, wh in PRINT_SIZES.items():
                thumb = _resize_fit(img, wh)
                buf = io.BytesIO()
                thumb.save(buf, "PNG")
                pv[name] = base64.b64encode(buf.getvalue()).decode("utf-8")
            previews_map[key] = pv

    if return_format.lower() == "zip":
        files_to_add: List[Tuple[str, bytes]] = []
        for r in results:
            if "image_b64" in r:
                files_to_add.append((f'{filename}_{r["style"]}.png', base64.b64decode(r["image_b64"])))
        if make_print_previews:
            for style, pv in previews_map.items():
                for name, b64s in pv.items():
                    files_to_add.append((f"{filename}_{style}_preview_{name}.png", base64.b64decode(b64s)))
        if not files_to_add:
            raise HTTPException(status_code=502, detail="Nothing to add to ZIP (no successful images).")

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in files_to_add:
                zf.writestr(name, data)
        zip_bytes = mem.getvalue()
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}.zip"',
            "Content-Length": str(len(zip_bytes)),
        }
        return Response(content=zip_bytes, media_type="application/zip", headers=headers)

    # default JSON
    return JSONResponse({
        "styles_requested": style_list,
        "canvas_size": [canvas.width, canvas.height],
        "art_bbox": keep_bbox,
        "normalize_ratio": normalize_ratio,
        "mat_pct": mat_pct,
        "api_size": "auto",
        "results": results,
        "previews": previews_map,
    })
