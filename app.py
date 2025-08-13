# app.py — Framed Wall-Art Mockups (deterministic compositing)
# Customer uploads full image -> choose "mockup type" -> place art into frame opening in real photo.
# No subject masking. Optional mat borders when aspect ratios don't match.

import io, os, json, base64, zipfile, glob, secrets
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response

# ---------------- App ----------------
app = FastAPI(
    title="Wall-Art Mockup API",
    description="Fits customer image into frame openings of photographic templates.",
    version="1.0.0",
)

# ---------------- Config ----------------
TEMPLATES_ROOT = os.getenv("TEMPLATES_ROOT", "templates")  # templates/<type>/<variant>/...
DEFAULT_VARIATIONS = int(os.getenv("DEFAULT_VARIATIONS", "5"))
DEFAULT_MAT_PCT = float(os.getenv("DEFAULT_MAT_PCT", "0.04"))  # 4% mat per side (of larger dim)
MAX_ART_SIDE = int(os.getenv("MAX_ART_SIDE", "3000"))          # clamp uploaded art for speed

# ---------------- Utils ----------------
def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _clamp(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return img.resize((int(w*s), int(h*s)), Image.LANCZOS)

def _add_mat(art: Image.Image, mat_pct: float, colour=(245, 245, 240)) -> Image.Image:
    if mat_pct <= 0:
        return art
    w, h = art.size
    pad = int(round(mat_pct * max(w, h)))
    canvas = Image.new("RGB", (w + 2*pad, h + 2*pad), colour)
    canvas.paste(art.convert("RGB"), (pad, pad))
    return canvas

def _fit_with_mat_to_aspect(img: Image.Image, target_aspect: float, mat_pct: float) -> Image.Image:
    """
    Ensures the *artwork area* (including mat) matches the frame opening aspect without cropping the art.
    We first letterbox/pillarbox with mat; then, if caller asked mat_pct>0 we add extra mat on top of that.
    """
    # letterbox with neutral mat to get the exact aspect
    w, h = img.size
    src_aspect = w / h
    if abs(src_aspect - target_aspect) < 1e-3:
        fitted = img
    elif src_aspect < target_aspect:
        # image too tall -> add left/right bars
        new_w = int(round(target_aspect * h))
        pad = (new_w - w) // 2
        base = Image.new("RGB", (new_w, h), (245, 245, 240))
        base.paste(img.convert("RGB"), (pad, 0))
        fitted = base
    else:
        # image too wide -> add top/bottom bars
        new_h = int(round(w / target_aspect))
        pad = (new_h - h) // 2
        base = Image.new("RGB", (w, new_h), (245, 245, 240))
        base.paste(img.convert("RGB"), (0, pad))
        fitted = base
    # optional extra mat to taste
    return _add_mat(fitted, mat_pct)

def _warp_into_quad(art_rgba: Image.Image, bg_rgba: Image.Image, quad: List[Tuple[float, float]]) -> Image.Image:
    """
    Perspective-warp 'art_rgba' into 'quad' over 'bg_rgba'.
    quad order: [top-left, top-right, bottom-right, bottom-left] — in background pixel coords.
    """
    bg = bg_rgba.convert("RGBA")
    a = art_rgba.convert("RGBA")
    w_src, h_src = a.size

    src = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]])
    dst = np.float32(quad)

    M = cv2.getPerspectiveTransform(src, dst)
    out_w, out_h = bg.size
    a_np = cv2.cvtColor(np.array(a), cv2.COLOR_RGBA2BGRA)
    warped = cv2.warpPerspective(a_np, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    warped_rgba = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA))
    comp = Image.alpha_composite(bg, warped_rgba)
    return comp

# ---------------- Template model ----------------
@dataclass
class Variant:
    key: str
    background_path: str
    overlay_path: Optional[str]        # frame rails + shadows + glass (with alpha)
    inner_quad: List[Tuple[float, float]]

@dataclass
class MockupType:
    type_key: str
    type_name: str
    variants: List[Variant]

def _load_variant(manifest_path: str) -> Variant:
    folder = os.path.dirname(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    quad = [(float(x), float(y)) for x, y in m["inner_quad"]]
    bg = os.path.join(folder, m["background"])
    ov = os.path.join(folder, m["overlay"]) if m.get("overlay") else None
    return Variant(key=m.get("key") or os.path.basename(folder), background_path=bg, overlay_path=ov, inner_quad=quad)

def _discover_types(root: str) -> List[MockupType]:
    """
    Expect structure:
      templates/<type>/<variant>/manifest.json
    Each manifest defines: inner_quad, background, overlay (optional)
    """
    types: Dict[str, List[Variant]] = {}
    for manifest in glob.glob(os.path.join(root, "*", "*", "manifest.json")):
        parts = manifest.replace("\\", "/").split("/")
        type_key = parts[-3]  # .../templates/<type>/<variant>/manifest.json
        v = _load_variant(manifest)
        types.setdefault(type_key, []).append(v)

    result: List[MockupType] = []
    for tkey, vars_ in types.items():
        # human name = title-case; can be overridden later if needed
        result.append(MockupType(type_key=tkey, type_name=tkey.replace("_", " ").title(), variants=sorted(vars_, key=lambda x: x.key)))
    return sorted(result, key=lambda t: t.type_key)

_TYPES_CACHE: Optional[List[MockupType]] = None
def _list_types() -> List[MockupType]:
    global _TYPES_CACHE
    if _TYPES_CACHE is None:
        _TYPES_CACHE = _discover_types(TEMPLATES_ROOT)
    return _TYPES_CACHE

# ---------------- Core compositor ----------------
def _compose_one(art_rgb: Image.Image, variant: Variant, mat_pct: float) -> Image.Image:
    # load BG and overlay
    bg = Image.open(variant.background_path).convert("RGBA")
    overlay = Image.open(variant.overlay_path).convert("RGBA") if variant.overlay_path else None

    # frame opening aspect
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = variant.inner_quad
    frame_w = ((x2 - x1) + (x3 - x4)) / 2.0
    frame_h = ((y4 - y1) + (y3 - y2)) / 2.0
    target_aspect = abs(frame_w / frame_h) if frame_w > 0 and frame_h > 0 else (art_rgb.width / art_rgb.height)

    # fit the art (do not crop) -> letterbox with neutral mat to match aspect, then optional extra mat
    art_prepped = _fit_with_mat_to_aspect(_clamp(art_rgb, MAX_ART_SIDE), target_aspect, mat_pct=mat_pct).convert("RGBA")

    # warp into the quad
    comp = _warp_into_quad(art_prepped, bg, variant.inner_quad)

    # add overlay (frame rails, bevel, glass, shadows)
    if overlay:
        comp = Image.alpha_composite(comp, overlay)

    return comp.convert("RGBA")

def _choose_variants(t: MockupType, n: int) -> List[Variant]:
    vars_ = t.variants
    if not vars_:
        raise HTTPException(status_code=500, detail=f"No variants found for type '{t.type_key}'")
    if n >= len(vars_):
        return vars_
    rng = secrets.SystemRandom()
    return [rng.choice(vars_) for _ in range(n)]

# ---------------- API ----------------
@app.get("/mockups/types")
def mockup_types():
    types = _list_types()
    return {
        "count": len(types),
        "types": [{"key": t.type_key, "name": t.type_name, "variants": len(t.variants)} for t in types]
    }

@app.post("/mockups/render/json")
async def mockups_render_json(
    file: UploadFile = File(...),
    type_key: str = Form(...),                 # e.g. "living_room"
    n: int = Form(DEFAULT_VARIATIONS),         # how many variations to return
    mat_pct: float = Form(DEFAULT_MAT_PCT),    # extra mat width % (after aspect fit)
):
    types = _list_types()
    match = next((t for t in types if t.type_key == type_key), None)
    if not match:
        raise HTTPException(status_code=400, detail=f"Unknown mockup type '{type_key}'. See /mockups/types")

    art = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results: List[Dict[str, Any]] = []
    for v in _choose_variants(match, n):
        out = _compose_one(art, v, mat_pct=mat_pct)
        results.append({"variant": v.key, "image_b64": _pil_to_b64_png(out)})
    return {"type": type_key, "count": len(results), "results": results}

@app.post("/mockups/render/html", response_class=HTMLResponse)
async def mockups_render_html(
    file: UploadFile = File(...),
    type_key: str = Form(...),
    n: int = Form(DEFAULT_VARIATIONS),
    mat_pct: float = Form(DEFAULT_MAT_PCT),
):
    js = await mockups_render_json(file=file, type_key=type_key, n=n, mat_pct=mat_pct)
    html = "<h1>Framed Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for r in js["results"]:
        html += f"<figure style='margin:8px'><img src='data:image/png;base64,{r['image_b64']}' style='width:360px;margin:5px;display:block'/><figcaption style='text-align:center;font:12px sans-serif'>{r['variant']}</figcaption></figure>"
    html += "</div>"
    return HTMLResponse(content=html)

@app.post("/mockups/render/zip")
async def mockups_render_zip(
    file: UploadFile = File(...),
    type_key: str = Form(...),
    n: int = Form(DEFAULT_VARIATIONS),
    mat_pct: float = Form(DEFAULT_MAT_PCT),
):
    js = await mockups_render_json(file=file, type_key=type_key, n=n, mat_pct=mat_pct)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(js["results"], 1):
            zf.writestr(f"{type_key}_{i:02d}_{r['variant']}.png", base64.b64decode(r["image_b64"]))
    mem.seek(0)
    headers = {"Content-Disposition": f"attachment; filename={type_key}_mockups.zip"}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)

# Simple 200 for health checks
@app.get("/healthz")
def healthz():
    return {"ok": True}
