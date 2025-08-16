# app.py — Outpainted Mockups (simple ingest resize + alignment-safe pipeline)
# 1) Ingest: optional proportional resize if long edge > ingest_max_long_edge (aspect ratio preserved)
# 2) Build canvas + mask from that exact raster
# 3) Quantize (canvas+mask) to OpenAI edit size (1024^2, 1024x1536, 1536x1024), call API
# 4) Scale result back to original canvas size
# 5) (Optional) Re-overlay original art with tiny inset to avoid seams

import io
import os
import base64
import zipfile
from math import ceil
from typing import Dict, Tuple, List

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Config / Presets
# =========================

DEFAULT_TARGET_PX = int(os.getenv("TARGET_PX", "2048"))          # upscale only if art is smaller
DEFAULT_INGEST_LONG_EDGE = int(os.getenv("INGEST_LONG_EDGE", "2048"))  # simple proportional downscale
OPENAI_MODEL = os.getenv("OPENAI_IMAGES_MODEL", "gpt-image-1")

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
        "Around the existing artwork, paint a refined study/home office: desaturated wall paint, "
        "subtle bookshelf blur and desk hints, muted daylight, thin dark metal frame, soft realistic shadows. No text."
    ),
    "gallery": (
        "Paint a clean gallery wall around the existing artwork: white wall with subtle microtexture, "
        "museum track lighting from above, thin black metal frame, natural falloff shadows. No text."
    ),
    "kitchen": (
        "Around the existing artwork, paint a tasteful kitchen nook: light painted wall, subtle tile/counter hints, "
        "soft daylight, thin black frame, realistic shadows. No text."
    ),
}
DEFAULT_STYLE_LIST = ["living_room", "bedroom", "study", "gallery", "kitchen"]

PRESERVE_DIRECTIVE = (
    "IMPORTANT: Preserve the existing artwork pixels exactly as-is within the KEEP region of the mask. "
    "Do not modify, blur, repaint, or regenerate any artwork pixels. "
    "Do not overlap the kept region with frame or mat; leave a clean thin gap; nothing should cover the art."
)

PRINT_SIZES: Dict[str, Tuple[int, int]] = {
    "4x5":   (1600, 2000),
    "3x4":   (1536, 2048),
    "2x3":   (1600, 2400),
    "11x14": (1650, 2100),
    "A4":    (1654, 2339),
}

# =========================
# FastAPI app
# =========================

app = FastAPI(title="Outpainted Mockups (simple ingest resize)", version="4.1")

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
        "ingest_long_edge_default": DEFAULT_INGEST_LONG_EDGE,
        "note": "Ingest proportional resize is ON by default (ingest_resize=1).",
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

def _ingest_simple_resize(file_bytes: bytes, enable: bool, max_long_edge: int) -> Image.Image:
    """
    Proportional resize ONLY if the long edge exceeds max_long_edge.
    No padding/cropping; preserves aspect ratio exactly.
    """
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    if not enable:
        return img.convert("RGBA")

    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_long_edge:
        s = max_long_edge / float(long_edge)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img.convert("RGBA")

def _pad_to_ratio(img: Image.Image, ratio_w: int, ratio_h: int, bg=(0, 0, 0, 0)):
    """Pad to exact ratio without scaling (letterbox), return (canvas, bbox_of_img)."""
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
    Then add border by pad_ratio; returns (canvas_rgba, art_bbox_on_canvas).
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
    OpenAI Images/edits semantics for gpt-image-1:
      transparent (alpha=0)  -> EDIT
      opaque     (alpha=255) -> KEEP
    We KEEP the artwork rectangle; EDIT everything else.
    """
    W, H = canvas_size
    x0, y0, x1, y1 = keep_bbox
    alpha = Image.new("L", (W, H), 0)               # EDIT outside
    keep  = Image.new("L", (x1 - x0, y1 - y0), 255) # KEEP inside
    alpha.paste(keep, (x0, y0))
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask.putalpha(alpha)
    return mask

def _overlay_original_art(result_rgba: Image.Image, art_rgba: Image.Image,
                          bbox: Tuple[int, int, int, int], inset_px: int = 0) -> Image.Image:
    """
    Paste original art back into the final image.
    If inset_px > 0, inset the destination rectangle to leave a thin margin (hides tiny misalignments).
    """
    x0, y0, x1, y1 = bbox
    if inset_px > 0:
        x0 += inset_px
        y0 += inset_px
        x1 -= inset_px
        y1 -= inset_px
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = bbox

    w, h = x1 - x0, y1 - y0
    if art_rgba.size != (w, h):
        art_rgba = art_rgba.resize((w, h), Image.LANCZOS)
    out = result_rgba.copy()
    out.paste(art_rgba, (x0, y0), art_rgba)
    return out

def _api_edit_size_for(canvas_size: Tuple[int, int]) -> Tuple[int, int, str]:
    """
    Pick OpenAI-supported edit size matching orientation:
      - Portrait  -> 1024x1536
      - Landscape -> 1536x1024
      - Square    -> 1024x1024
    """
    W, H = canvas_size
    if W == H:
        return 1024, 1024, "1024x1024"
    if H > W:
        return 1024, 1536, "1024x1536"
    return 1536, 1024, "1536x1024"

def _openai_images_edit_multi(image_png: bytes, mask_png: bytes, prompt: str, n: int, size_str: str) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        headers["OpenAI-Organization"] = org_id

    data = {"model": OPENAI_MODEL, "prompt": prompt, "size": size_str, "n": str(max(1, min(int(n), 10)))}
    files = {
        "image": ("canvas.png", image_png, "image/png"),
        "mask":  ("mask.png",   mask_png,  "image/png"),
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


# =========================================================
# Single-style endpoint
# =========================================================
@app.post("/outpaint/mockup_single")
async def outpaint_mockup_single(
    file: UploadFile = File(...),
    style: str = Form("living_room"),
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    normalize_ratio: str = Form(""),
    mat_pct: float = Form(0.0),
    overlay_original: int = Form(0),          # default OFF to avoid seams
    overlay_inset_px: int = Form(0),          # if overlay=1, set 1–3 to hide frame misalignment
    # ingest proportional resize (simple & safe):
    ingest_resize: int = Form(1),             # 💡 ON by default
    ingest_max_long_edge: int = Form(DEFAULT_INGEST_LONG_EDGE),
    return_format: str = Form("png"),
    filename: str = Form("mockup_single"),
):
    if style not in STYLE_PROMPTS:
        raise HTTPException(400, f"Unknown style '{style}'. Choose from {list(STYLE_PROMPTS.keys())}")

    try:
        raw = await file.read()
        art = _ingest_simple_resize(raw, bool(ingest_resize), int(ingest_max_long_edge))
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
        mx = int(w * mat_pct / 2.0); my = int(h * mat_pct / 2.0)
        mat_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mat_canvas.paste(art, (mx, my), art)
        art = mat_canvas

    # Build canvas + mask from the same raster
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    # Quantize to OpenAI size together
    api_w, api_h, api_size_str = _api_edit_size_for(canvas.size)
    canvas_api = canvas.resize((api_w, api_h), Image.LANCZOS)
    mask_api   = mask.resize((api_w, api_h), Image.NEAREST)

    prompt = f"{PRESERVE_DIRECTIVE} {STYLE_PROMPTS[style]}"
    b64 = _openai_images_edit_multi(
        _img_to_png_bytes(canvas_api),
        _img_to_png_bytes(mask_api),
        prompt=prompt,
        n=1,
        size_str=api_size_str,
    )[0]

    out = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    if out.size != canvas.size:
        out = out.resize(canvas.size, Image.LANCZOS)

    if overlay_original:
        x0, y0, x1, y1 = keep_bbox
        placed_art = canvas.crop((x0, y0, x1, y1))
        out = _overlay_original_art(out, placed_art, keep_bbox, inset_px=max(0, int(overlay_inset_px)))

    if return_format.lower() == "json":
        buf = io.BytesIO(); out.save(buf, "PNG")
        return JSONResponse({
            "style": style,
            "canvas_size": [canvas.width, canvas.height],
            "art_bbox": keep_bbox,
            "normalize_ratio": normalize_ratio,
            "mat_pct": mat_pct,
            "api_size": api_size_str,
            "ingest_resize": int(ingest_resize),
            "ingest_max_long_edge": int(ingest_max_long_edge),
            "image_b64": base64.b64encode(buf.getvalue()).decode("utf-8")
        })

    buf = io.BytesIO(); out.save(buf, "PNG")
    headers = {"Content-Disposition": f'inline; filename="{filename}_{style}.png"'}
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

# =========================================================
# Multi-style / multi-variant endpoint
# =========================================================
@app.post("/outpaint/mockup")
async def outpaint_mockup(
    file: UploadFile = File(...),
    styles: str = Form(",".join(DEFAULT_STYLE_LIST)),
    target_px: int = Form(DEFAULT_TARGET_PX),
    pad_ratio: float = Form(0.42),
    normalize_ratio: str = Form(""),
    mat_pct: float = Form(0.0),
    variants: int = Form(5),
    overlay_original: int = Form(0),
    overlay_inset_px: int = Form(0),
    make_print_previews: int = Form(0),
    # ingest proportional resize (simple & safe):
    ingest_resize: int = Form(1),             # 💡 ON by default
    ingest_max_long_edge: int = Form(DEFAULT_INGEST_LONG_EDGE),
    return_format: str = Form("json"),
    filename: str = Form("mockup_bundle"),
):
    style_list = [s.strip() for s in styles.split(",") if s.strip()]
    invalid = [s for s in style_list if s not in STYLE_PROMPTS]
    if invalid:
        raise HTTPException(400, f"Unknown styles {invalid}. Choose from {list(STYLE_PROMPTS.keys())}")
    if not style_list:
        raise HTTPException(400, "No valid styles provided.")

    try:
        raw = await file.read()
        art = _ingest_simple_resize(raw, bool(ingest_resize), int(ingest_max_long_edge))
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
        mx = int(w * mat_pct / 2.0); my = int(h * mat_pct / 2.0)
        mat_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mat_canvas.paste(art, (mx, my), art)
        art = mat_canvas

    # Build canvas + mask (one geometry)
    canvas, keep_bbox = _pad_canvas_keep_center(art, pad_ratio=pad_ratio, target_side=target_px)
    mask = _build_outpaint_mask(canvas.size, keep_bbox)

    # API-safe size
    api_w, api_h, api_size_str = _api_edit_size_for(canvas.size)
    canvas_api = canvas.resize((api_w, api_h), Image.LANCZOS)
    mask_api   = mask.resize((api_w, api_h), Image.NEAREST)

    one_style = len(style_list) == 1
    n_per_style = max(1, min(int(variants), 10)) if one_style else 1

    # Fast PNG path → first variant, first style
    if return_format.lower() == "png":
        style = style_list[0]
        prompt = f"{PRESERVE_DIRECTIVE} {STYLE_PROMPTS[style]}"
        b64_list = _openai_images_edit_multi(
            _img_to_png_bytes(canvas_api), _img_to_png_bytes(mask_api),
            prompt=prompt, n=n_per_style, size_str=api_size_str
        )
        out = Image.open(io.BytesIO(base64.b64decode(b64_list[0]))).convert("RGBA")
        if out.size != canvas.size:
            out = out.resize(canvas.size, Image.LANCZOS)
        if overlay_original:
            x0, y0, x1, y1 = keep_bbox
            placed_art = canvas.crop((x0, y0, x1, y1))
            out = _overlay_original_art(out, placed_art, keep_bbox, inset_px=max(0, int(overlay_inset_px)))
        buf = io.BytesIO(); out.save(buf, "PNG")
        suffix = "_v01" if n_per_style > 1 else ""
        headers = {"Content-Disposition": f'inline; filename="{filename}_{style}{suffix}.png"'}
        return Response(content=buf.getvalue(), media_type="image/png", headers=headers)

    results: List[Dict[str, object]] = []
    previews_map: Dict[str, Dict[str, Dict[str, str]]] = {}

    for style in style_list:
        prompt = f"{PRESERVE_DIRECTIVE} {STYLE_PROMPTS[style]}"
        b64_list = _openai_images_edit_multi(
            _img_to_png_bytes(canvas_api), _img_to_png_bytes(mask_api),
            prompt=prompt, n=n_per_style, size_str=api_size_str
        )

        fixed_b64_list: List[str] = []
        for b64 in b64_list:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
            if img.size != canvas.size:
                img = img.resize(canvas.size, Image.LANCZOS)
            if overlay_original:
                x0, y0, x1, y1 = keep_bbox
                placed_art = canvas.crop((x0, y0, x1, y1))
                img = _overlay_original_art(img, placed_art, keep_bbox, inset_px=max(0, int(overlay_inset_px)))
            buf = io.BytesIO(); img.save(buf, "PNG")
            fixed_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        results.append({
            "style": style,
            "api_size": api_size_str,
            "ingest_resize": int(ingest_resize),
            "ingest_max_long_edge": int(ingest_max_long_edge),
            "image_b64": fixed_b64_list[0],
            "variants": fixed_b64_list,
        })

        if make_print_previews:
            pv_style: Dict[str, Dict[str, str]] = {}
            for i, vb64 in enumerate(fixed_b64_list, start=1):
                img = Image.open(io.BytesIO(base64.b64decode(vb64))).convert("RGBA")
                pv_variant: Dict[str, str] = {}
                for name, wh in PRINT_SIZES.items():
                    thumb = _resize_fit(img, wh)
                    tbuf = io.BytesIO(); thumb.save(tbuf, "PNG")
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
                    for vkey, pv_set in pv.items():
                        for name, b64s in pv_set.items():
                            zf.writestr(f"{filename}_{style}_{vkey}_preview_{name}.png", base64.b64decode(b64s))
        zip_bytes = mem.getvalue()
        headers = {"Content-Disposition": f'attachment; filename="{filename}.zip"',
                   "Content-Length": str(len(zip_bytes))}
        return Response(content=zip_bytes, media_type="application/zip", headers=headers)

    return JSONResponse({
        "styles_requested": style_list,
        "variants_per_style": n_per_style,
        "canvas_size": [canvas.width, canvas.height],
        "art_bbox": keep_bbox,
        "normalize_ratio": normalize_ratio,
        "mat_pct": mat_pct,
        "api_size": api_size_str,
        "ingest_resize": int(ingest_resize),
        "ingest_max_long_edge": int(ingest_max_long_edge),
        "results": results,
        "previews": previews_map,
    })
