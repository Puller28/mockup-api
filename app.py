import os
import json
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from io import BytesIO
from typing import List
from PIL import Image

# -------------------------------
# App Setup
# -------------------------------
app = FastAPI(
    title="Perspective Mockup API",
    version="1.0",
    docs_url="/docs",         # Swagger UI
    redoc_url="/redoc",       # ReDoc UI (optional)
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

TEMPLATE_ROOT = Path("templates")

# -------------------------------
# Utilities
# -------------------------------
def load_manifest(template_dir: Path):
    manifest_file = template_dir / "manifest"
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            return json.load(f)
    return {"templates": []}


def warp_perspective(art_img: np.ndarray, target_quad: List[List[int]], fit="contain"):
    """ Warp artwork into the target quadrilateral """
    h, w = art_img.shape[:2]
    target_quad = np.array(target_quad, dtype=np.float32)

    # Source points = full rectangle of art
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Homography
    matrix = cv2.getPerspectiveTransform(src, target_quad)

    # Output canvas size = bounding box of quad
    max_x = int(target_quad[:, 0].max())
    max_y = int(target_quad[:, 1].max())

    warped = cv2.warpPerspective(art_img, matrix, (max_x, max_y))
    return warped


def overlay(base_img: np.ndarray, overlay_img: np.ndarray, opacity=1.0):
    """ Blend overlay onto base image with alpha/opacity """
    y1, x1 = 0, 0
    y2, x2 = overlay_img.shape[0], overlay_img.shape[1]

    roi = base_img[y1:y2, x1:x2]

    # If overlay has alpha
    if overlay_img.shape[2] == 4:
        alpha_overlay = overlay_img[:, :, 3] / 255.0
        alpha_overlay = np.expand_dims(alpha_overlay, axis=2)
        alpha_base = 1.0 - alpha_overlay

        for c in range(0, 3):
            roi[:, :, c] = (
                alpha_base[:, :, 0] * roi[:, :, c] +
                alpha_overlay[:, :, 0] * overlay_img[:, :, c] * opacity
            )
    else:
        roi[:] = cv2.addWeighted(roi, 1 - opacity, overlay_img, opacity, 0)

    base_img[y1:y2, x1:x2] = roi
    return base_img


# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/templates", tags=["templates"], summary="List available templates")
def list_templates():
    """Return all available template rooms and frame definitions"""
    results = {}
    for room_dir in TEMPLATE_ROOT.iterdir():
        if room_dir.is_dir():
            manifest = load_manifest(room_dir)
            results[room_dir.name] = manifest.get("templates", [])
    return results


@app.post(
    "/mockup/apply",
    tags=["mockup"],
    summary="Apply artwork into a room template",
    description=(
        "Upload an artwork image and place it into a chosen template using a perspective warp.\n\n"
        "**Form fields**\n"
        "- `file` (required): the artwork image (PNG/JPG)\n"
        "- `room` (required): e.g. `living_room`\n"
        "- `template_id` (optional): if omitted, the first template in that room is used\n"
        "- `fit`: `contain` (default) or `cover`\n"
        "- `margin_px`: inset the frame area (default 0)\n"
        "- `feather_px`: soft edge blend (override manifest)\n"
        "- `opacity`: 0–1 (override manifest)\n"
        "- `return_format`: `png` (default) or `json`"
    )
)
async def mockup_apply(
    file: UploadFile = File(..., description="Artwork image (PNG/JPG)"),
    room: str = Form(..., description="Room folder name under templates/, e.g. 'living_room'"),
    template_id: str = Form("", description="Template `id` from /templates (optional)"),
    fit: str = Form("contain", description="How to fit art into frame: 'contain' or 'cover'"),
    margin_px: int = Form(0, description="Inset inside frame in pixels (default 0)"),
    feather_px: float = Form(-1.0, description="Override manifest feather; -1 = use manifest"),
    opacity: float = Form(-1.0, description="Override opacity 0..1; -1 = use manifest"),
    return_format: str = Form("png", description="png | json (base64)")
):
    # --- Load template ---
    template_dir = TEMPLATE_ROOT / room
    manifest = load_manifest(template_dir)

    if not manifest.get("templates"):
        return JSONResponse({"error": "No templates defined for this room"}, status_code=400)

    template = manifest["templates"][0] if not template_id else next(
        (t for t in manifest["templates"] if t["id"] == template_id),
        manifest["templates"][0]
    )

    bg_path = template_dir / template["background"]
    if not bg_path.exists():
        return JSONResponse({"error": f"Background not found: {bg_path}"}, status_code=400)

    # --- Load images ---
    bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    art_pil = Image.open(BytesIO(await file.read())).convert("RGBA")
    art = cv2.cvtColor(np.array(art_pil), cv2.COLOR_RGBA2BGRA)

    # --- Warp ---
    warped = warp_perspective(art, template["frame_quad"], fit=fit)

    # --- Composite ---
    out = overlay(bg, warped, opacity=1.0 if opacity < 0 else opacity)

    # --- Return ---
    out_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)

    if return_format == "json":
        import base64
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": b64}

    return StreamingResponse(buf, media_type="image/png")
