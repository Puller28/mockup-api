# app.py
import os
import io
import json
import base64
from typing import List, Dict

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from PIL import Image, ImageDraw

# ================== ENV ==================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")  # e.g. https://api.runpod.ai/v2/<endpoint_id>
DEFAULT_CKPT = os.getenv("CKPT_NAME", "flux1-dev-fp8.safetensors")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
    raise RuntimeError("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT env vars on Render.")

# ================== PROMPTS ==================
TEMPLATES: Dict[str, str] = {
    "bedroom": "Framed artwork hanging in a cozy bedroom with sunlight filtering through linen curtains, photorealistic interior, soft natural light, realistic shadows, DSLR photo, realistic wall texture, subtle reflections on frame glass.",
    "gallery_wall": "Framed print on a gallery wall with spot lighting and minimal decor, photorealistic, clean plaster wall, realistic shadows from frame, subtle gallery ambience.",
    "modern_lounge": "Framed artwork in a modern minimalist lounge above a sofa, natural window light, neutral palette, photorealistic, realistic soft shadows, clean architecture.",
    "rustic_study": "Framed artwork in a rustic study with wooden shelves and warm desk lamp, photorealistic, cozy lighting, natural wood textures, believable shadows.",
    "kitchen": "Framed botanical print in a bright modern kitchen with plants, daylight, photorealistic, subtle reflections and realistic shadows."
}
NEGATIVE_PROMPT = (
    "blurry, low detail, distorted, bad framing, artifacts, low quality, overexposed, "
    "underexposed, warped perspective, extra objects, text, watermark, logo"
)

# ================== FASTAPI ==================
app = FastAPI(title="Mockup Outpainting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class BatchResponse(BaseModel):
    template: str
    prompt: str
    images: List[str]  # data-URLs or http URLs

# ================== HTTP SESSION ==================
def _build_session() -> requests.Session:
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

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

# ================== HELPERS ==================
def to_data_url(b64_png_or_url: str) -> str:
    if b64_png_or_url.startswith("http://") or b64_png_or_url.startswith("https://"):
        return b64_png_or_url
    if b64_png_or_url.startswith("data:image/"):
        return b64_png_or_url
    return "data:image/png;base64," + b64_png_or_url

def extract_images_from_output(status_payload: dict) -> List[str]:
    """
    Normalize RunPod/ComfyUI outputs to a list of displayable strings.
    Supports keys: url, base64, content, data, image, urls, image_url.
    """
    out = (status_payload or {}).get("output") or {}
    results: List[str] = []

    imgs = out.get("images")
    if isinstance(imgs, list):
        for it in imgs:
            if isinstance(it, dict):
                if it.get("url"):
                    results.append(it["url"])
                elif it.get("base64"):
                    results.append(to_data_url(it["base64"]))
                elif it.get("content"):
                    results.append(to_data_url(it["content"]))
                elif it.get("data"):
                    results.append(to_data_url(it["data"]))
                elif it.get("image"):
                    results.append(to_data_url(it["image"]))
            elif isinstance(it, str):
                results.append(to_data_url(it))
        if results:
            return results

    urls = out.get("urls")
    if isinstance(urls, list) and urls:
        return urls

    if isinstance(out.get("image_url"), str) and out["image_url"]:
        return [out["image_url"]]

    b64s = out.get("base64")
    if isinstance(b64s, list) and b64s:
        return [to_data_url(b) for b in b64s if isinstance(b, str) and b]

    data_arr = out.get("data")
    if isinstance(data_arr, list):
        for item in data_arr:
            if isinstance(item, dict) and isinstance(item.get("images"), list):
                for it in item["images"]:
                    if isinstance(it, dict):
                        if it.get("url"):
                            results.append(it["url"])
                        elif it.get("base64"):
                            results.append(to_data_url(it["base64"]))
                        elif it.get("content"):
                            results.append(to_data_url(it["content"]))
                        elif it.get("data"):
                            results.append(to_data_url(it["data"]))
                    elif isinstance(it, str):
                        results.append(to_data_url(it))
        if results:
            return results

    return results

def call_runsync(payload: dict, timeout_sec: int = 480) -> dict:
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

# ================== OUTPAINT PREP ==================
CANVAS_SIZE = 1024                # final square image
ART_MAX_W, ART_MAX_H = 768, 896   # art box (keep aspect), leaves margins for AI to paint
CANVAS_BG = (210, 210, 210)       # neutral mid-light gray

def make_canvas_and_mask(art_bytes: bytes) -> Dict[str, str]:
    """
    Build:
      - init.png (1024x1024) with the artwork centered (untouched)
      - mask.png (white outside art, black on art)
    Return base64 PNGs (no data: prefix) + frame box info (for logging/debug).
    """
    art = Image.open(io.BytesIO(art_bytes)).convert("RGB")
    aw, ah = art.size
    scale = min(ART_MAX_W / aw, ART_MAX_H / ah, 1.0)
    tw, th = max(1, int(aw * scale)), max(1, int(ah * scale))

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CANVAS_BG)
    x = (CANVAS_SIZE - tw) // 2
    y = (CANVAS_SIZE - th) // 2
    art_resized = art.resize((tw, th), Image.LANCZOS)
    canvas.paste(art_resized, (x, y))

    # Mask: white (255)=paint here, black (0)=protect artwork
    mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + tw, y + th], fill=0)

    def to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "init_png_b64": to_b64(canvas),
        "mask_png_b64": to_b64(mask),
        "frame_box": json.dumps({"x": x, "y": y, "w": tw, "h": th})
    }

# ================== WORKFLOW (INPAINT OUTSIDE ART) ==================
def build_outpaint_workflow(prompt: str, neg: str, seed: int,
                            init_b64_png: str, mask_b64_png: str) -> dict:
    """
    True outpainting:
      - Load init canvas (art centered)
      - Load mask image, convert IMAGE->MASK with channel="luminance"
      - VAEEncodeForInpaint with grow_mask_by
      - KSampler generates ONLY outside mask
      - Decode + Save
    """
    workflow = {
        "100": {  # Model
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": DEFAULT_CKPT}
        },

        # Text conditioning
        "3": { "class_type": "CLIPTextEncode", "inputs": {"clip": ["100", 1], "text": prompt} },
        "4": { "class_type": "CLIPTextEncode", "inputs": {"clip": ["100", 1], "text": neg} },

        # Images
        "10": { "class_type": "LoadImage", "inputs": {"image": "init.png", "upload": True} },
        "11": { "class_type": "LoadImage", "inputs": {"image": "mask.png", "upload": True} },
        "11m": {  # convert IMAGE -> MASK
    "class_type": "ImageToMask",
    "inputs": {
        "image": ["11", 0],
        "channel": "red"   # was "luminance"; allowed: red|green|blue|alpha
    }
},


        # Encode for inpaint (requires MASK + grow_mask_by)
        "12": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["10", 0],
                "mask": ["11m", 0],
                "grow_mask_by": 8,  # expand protection (tune 4–16)
                "vae": ["100", 2]
            }
        },

        # KSampler paints ONLY outside the mask
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["100", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["12", 0],
                "seed": seed,
                "steps": 28,
                "cfg": 6.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.70
            }
        },

        # Decode & Save
        "18": { "class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["100", 2]} },
        "19": { "class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "outpainted_mockup"} }
    }

    return {
        "workflow": workflow,
        "images": [
            {"name": "init.png", "image": "data:image/png;base64," + init_b64_png},
            {"name": "mask.png", "image": "data:image/png;base64," + mask_b64_png}
        ]
    }

# ================== ROUTES ==================
@app.get("/")
def root():
    return {"message": "Mockup Outpainting API running"}

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

@app.post("/batch", response_model=BatchResponse)
async def batch(template: str = Form(...), file: UploadFile = File(...)):
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Invalid template. Available: {list(TEMPLATES.keys())}")

    art_bytes = await file.read()
    prep = make_canvas_and_mask(art_bytes)
    init_b64 = prep["init_png_b64"]
    mask_b64 = prep["mask_png_b64"]

    prompt_text = TEMPLATES[template]
    images_all: List[str] = []

    for i in range(5):
        wf_input = build_outpaint_workflow(
            prompt=prompt_text,
            neg=NEGATIVE_PROMPT,
            seed=1234567 + i,
            init_b64_png=init_b64,
            mask_b64_png=mask_b64
        )
        payload = {"input": {"return_type": "base64", **wf_input}}
        result = call_runsync(payload, timeout_sec=480)

        if i == 0:
            try:
                print("RUNPOD_RAW_SAMPLE:", json.dumps(result)[:4000])
            except Exception:
                print("RUNPOD_RAW_SAMPLE: <non-serializable>")

        outs = extract_images_from_output(result)
        images_all.append(outs[0] if outs else "MISSING")

    return BatchResponse(template=template, prompt=prompt_text, images=images_all)

@app.post("/batch/html", response_class=HTMLResponse)
async def batch_html(template: str = Form(...), file: UploadFile = File(...)):
    resp = await batch(template, file)  # reuse logic
    html_imgs = "".join(
        f'<div style="margin:10px 0"><img style="max-width:640px" src="{u}"><br>'
        f'<small>{u[:88]}...</small></div>' for u in resp.images
    )
    return f"<h3>{resp.template}</h3><p>{resp.prompt}</p>{html_imgs}"
