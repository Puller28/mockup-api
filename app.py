# app.py — Outpainted Wall-Art Mockups
# - Strict preserve (mask + re-overlay original)
# - Single style => N variants; multiple styles => 1 each
# - Return: json | png | zip
# - Utility endpoint /utils/fit_under_1mb to shrink uploads before mockups
# - No changes required to your caller unless you want to use the resizer

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

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Config / Presets
# =========================

DEFAULT_TARGET_PX = int(os.getenv("TARGET_PX", "2048"))  # only upscales if smaller

PRINT_SIZES: Dict[str, Tuple[int, int]] = {
    "4x5":   (1600, 2000),
    "3x4":   (1536, 2048),
    "2x3":   (1600, 2400),
    "11x14": (1650, 2100),
    "A4":    (1654, 2339),
}

STYLE_PROMPTS: Dict[str, str] = {
    "living_room": (
        "Create a realistic interior mockup AROUND the existing artwork. "
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

PRESERVE_DIRECTIVE = (
    "IMPORTANT: Preserve the existing artwork pixels exactly as-is within the KEEP region of the mask. "
    "Do not modify, blur, repaint, recompose, or regenerate ANY part of the artwork. "
    "Only generate the surrounding wall, frame, lighting and room context OUTSIDE the artwork region."
)

# =========================
# FastAPI app
# =========================

app = FastAPI(title="Outpainted Mockups (strict preserve + variants + resizer)", version="3.3")

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
        "note": "Uploads are used full-size. For big files, call /utils/fit_under_1mb first.",
    }

# =========================
# Helpers
# =========================

def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def _img_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _resize_fit(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    return ImageOps.contain(img, target, Image.LANCZOS)

def _ingest_user_image_fullsize(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def _pad_to_ratio(img: Image.Image, ratio_w: int, ratio_h: int, bg=(0, 0, 0, 0)):
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
    """
    Upscale to target_side (max edge) ONLY if image is smaller; never downscale.
    Then add border by pad_ratio; returns (canvas, art_bbox).
    """
    img = img.convert("RGBA")
    longest = max(img.size)
    if longest < target_side:
        scale = target_side / float(longest)
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
    OpenAI Images/edits semantics:
      transparent (alpha=0)  -> EDIT
      opaque     (alpha=255) -> KEEP
    We create a binary mask: KEEP artwork, EDIT everything else.
    """
    W, H = canvas_size
    x0, y0, x1, y1 = keep_bbox
    alpha = Image.new("L", (W, H), 0)               # EDIT outside
    keep  = Image.new("L", (x1 - x0, y1 - y0), 255) # KEEP inside
    alpha.paste(keep, (x0, y0))
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask.putalpha(alpha)
    return mask

def _overlay_original_art(result_rgba: Image.Image, art_rgba: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Guarantee pixel-perfect preservation by pasting the original art back into the result."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if art_rgba.size != (w, h):
        art_rgba = art_rgba.resize((w, h), Image.LANCZOS)
    out = result_rgba.copy()
    out.paste(art_rgba, (x0, y0), art_rgba)
    return out

def _openai_images_edit_multi(image_bytes: bytes, mask_bytes: bytes, prompt: str, n: int = 1, size: str = "auto") -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        headers["OpenAI-Organization"] = org_id
    data = {"model": "gpt-image-1", "prompt": prompt, "size": size, "n": str(max(1, min(int(n), 10)))}
    files = {
        "image": ("canvas.png", image_bytes, "image/png"),
        "mask":  ("mask.png",   mask_bytes,  "image/png"),
    }
    try:
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=300)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Images API request failed: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Image API error [{resp.status_code}]: {resp.text}")
    js = resp.json()
    items = js.get("data") or []
    if not items:
        raise HTTPException(status_code=502, detail=f"Image API returned no data: {js}")
    return [it.get("b64_json") for it in items if it.get("b64_json")]

# =========================
# Utility: pre-shrink to <= 1 MB (or custom)
# =========================
@app.post("/utils/fit_under_1mb")
async def fit_under_1mb(
    file: UploadFile = File(...),
    max_bytes: int = Form(1_000_000),
    max_side: int = Form(4096),           # optional ceiling
    format: str = Form("jpeg"),           # "jpeg" (recommended) or "png"
    min_quality: int = Form(60),          # for JPEG search
):
    """
    Returns a size-reduced image <= max_bytes, preserving aspect ratio.
    Strategy: try current size -> JPEG quality binary search -> if still too big, downscale in steps.
    """
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    if max(img.size) > max_side:
        s = max_side / float(max(img.size))
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)

    if format.lower() == "png":
        cur = _img_to_png_bytes(img)
        while len(cur) > max_bytes and max(img.size) > 512:
            img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.LANCZOS)
            cur = _img_to_png_bytes(img)
        out_b = cur
        mime = "image/png"
        ext = "png"
    else:
        lo, hi = min_quality, 95
        best = _img_to_jpeg_bytes(img, hi)
        if len(best) > max_bytes:
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = _img_to_jpeg_bytes(img, mid)
                if len(cand) <= max_bytes:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
        while len(best) > max_bytes and max(img.size) > 512:
            img = img.resize((int(img.width * 0.92), int(img.height * 0.92)), Image.LANCZOS)
            lo, hi = min_quality, 90
            best = _img_to_jpeg_bytes(img, hi)
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = _img_to_jpeg_bytes(img, mid)
                if len(cand) <= max_bytes:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
        out_b = best
        mime = "image/jpeg"
        ext = "jpg"

    b64 = base64.b64encode(out_b).decode("utf-8")
    return JSONResponse({
        "format": ext,
        "mime": mime,
        "size_bytes": len(out_b),
        "width": img.width,
        "height": img.height,
        "image_b64": b64
    })

# =========================================================
# Single-style endpoint (adds overlay_original)
# =========================================================
@app.post("/outpaint/mockup_single")
async def outpaint_mockup_single(
    file: UploadFile = File(...),
    style: str = Form("living_room"),
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    normalize_ratio: str = Form(""),
    mat_pct: float = Form(0.0),
    overlay_original: int = Form(1),          # paste back original art (default on)
    return_format: str = Form("png"),         # "png" | "json"
    filename: str = Form("mockup_single"),
):
    if style not in STYLE_PROMPTS:
        raise HTTPException(400, f"Unknown style '{style}'. Choose from {list(STYLE_PROMPTS.keys())}")

    try:
        art = _ingest_user_image_fullsize(await file.read())
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

    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    prompt = f"{PRESERVE_DIRECTIVE} {STYLE_PROMPTS[style]}"
    b64 = _openai_images_edit_multi(_img_to_png_bytes(canvas), _img_to_png_bytes(mask),
                                    prompt=prompt, n=1, size="auto")[0]
    out = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

    if overlay_original:
        x0, y0, x1, y1 = keep_bbox
        placed_art = canvas.crop((x0, y0, x1, y1))
        out = _overlay_original_art(out, placed_art, keep_bbox)

    if return_format.lower() == "json":
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return JSONResponse({
            "style": style,
            "canvas_size": [canvas.width, canvas.height],
            "art_bbox": keep_bbox,
            "normalize_ratio": normalize_ratio,
            "mat_pct": mat_pct,
            "api_size": "auto",
            "image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")
        })

    buf = io.BytesIO()
    out.save(buf, "PNG")
    headers = {"Content-Disposition": f'inline; filename="{filename}_{style}.png"'}
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

# =========================================================
# Multi-variant endpoint (single style -> N variants). Adds overlay_original.
# =========================================================
@app.post("/outpaint/mockup")
async def outpaint_mockup(
    file: UploadFile = File(...),
    styles: str = Form(",".join(DEFAULT_STYLE_LIST)),   # comma-separated list
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    normalize_ratio: str = Form(""),
    mat_pct: float = Form(0.0),
    variants: int = Form(5),                            # used only when exactly one style is provided
    overlay_original: int = Form(1),                    # paste back original art
    make_print_previews: int = Form(0),                 # optional; heavy when many variants
    return_format: str = Form("json"),                  # "json" | "png" | "zip"
    filename: str = Form("mockup_bundle"),
):
    style_list = [s.strip() for s in styles.split(",") if s.strip()]
    invalid = [s for s in style_list if s not in STYLE_PROMPTS]
    if invalid:
        raise HTTPException(400, f"Unknown styles {invalid}. Choose from {list(STYLE_PROMPTS.keys())}")
    if not style_list:
        raise HTTPException(400, "No valid styles provided.")

    try:
        art = _ingest_user_image_fullsize(await file.read())
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

    # Canvas/mask once
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)
    canvas_bytes = _img_to_png_bytes(canvas)
    mask_bytes   = _img_to_png_bytes(mask)

    one_style = len(style_list) == 1
    n_per_style = max(1, min(int(variants), 10)) if one_style else 1

    # PNG fast path → just return first image
    if return_format.lower() == "png":
        style = style_list[0]
        prompt = f"{PRESERVE_DIRECTIVE} {STYLE_PROMPTS[style]}"
        b64_list = _openai_images_edit_multi(canvas_bytes, mask_bytes, prompt=prompt, n=n_per_style, size="auto")
        out = Image.open(io.BytesIO(base64.b64decode(b64_list[0]))).convert("RGBA")
        if overlay_original:
            x0, y0, x1, y1 = keep_bbox
            placed_art = canvas.crop((x0, y0, x1, y1))
            out = _overlay_original_art(out, placed_art, keep_bbox)
        buf = io.BytesIO()
        out.save(buf, "PNG")
        suffix = "_v01" if n_per_style > 1 else ""
        headers = {"Content-Disposition": f'inline; filename="{filename}_{style}{suffix}.png"'}
        return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

    # JSON / ZIP aggregate
    results: List[Dict[str, object]] = []
    previews_map: Dict[str, Dict[str, Dict[str, str]]] = {}

    for style in style_list:
        prompt = f"{PRESERVE_DIRECTIVE} {STYLE_PROMPTS[style]}"
        b64_list = _openai_images_edit_multi(canvas_bytes, mask_bytes, prompt=prompt, n=n_per_style, size="auto")

        # Apply overlay per variant if requested
        fixed_b64_list: List[str] = []
        for b64 in b64_list:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
            if overlay_original:
                x0, y0, x1, y1 = keep_bbox
                placed_art = canvas.crop((x0, y0, x1, y1))
                img = _overlay_original_art(img, placed_art, keep_bbox)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            fixed_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        results.append({
            "style": style,
            "image_b64": fixed_b64_list[0],
            "variants": fixed_b64_list,   # list of base64 strings
        })

        if make_print_previews:
            pv_style: Dict[str, Dict[str, str]] = {}
            for i, vb64 in enumerate(fixed_b64_list, start=1):
                img = Image.open(io.BytesIO(base64.b64decode(vb64))).convert("RGBA")
                pv_variant: Dict[str, str] = {}
                for name, wh in PRINT_SIZES.items():
                    thumb = _resize_fit(img, wh)
                    tbuf = io.BytesIO()
                    thumb.save(tbuf, "PNG")
                    pv_variant[name] = base64.b64encode(tbuf.getvalue()).decode("utf-8")
                pv_style[f"v{i:02d}"] = pv_variant
            previews_map[style] = pv_style

    if return_format.lower() == "zip":
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in results:
                style = item["style"]
                for i, b64 in enumerate(item["variants"], start=1):
                    zf.writestr(f"{filename}_{style}_v{i:02d}.png", base64.b64decode(b64))
            if make_print_previews:
                for style, pv in previews_map.items():
                    for vkey, pv_set in pv.items():  # v01, v02, ...
                        for name, b64s in pv_set.items():
                            zf.writestr(f"{filename}_{style}_{vkey}_preview_{name}.png", base64.b64decode(b64s))
        zip_bytes = mem.getvalue()
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}.zip"',
            "Content-Length": str(len(zip_bytes)),
        }
        return Response(content=zip_bytes, media_type="application/zip", headers=headers)

    # default JSON
    return JSONResponse({
        "styles_requested": style_list,
        "variants_per_style": n_per_style,
        "canvas_size": [canvas.width, canvas.height],
        "art_bbox": keep_bbox,
        "normalize_ratio": normalize_ratio,
        "mat_pct": mat_pct,
        "api_size": "auto",
        "results": results,          # each has {"style", "image_b64", "variants":[...]}
        "previews": previews_map,    # if requested
    })
