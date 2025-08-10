import os
import io
import gc
import json
import base64
import secrets
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps, ImageFilter
from rembg import remove, new_session
import requests


# -------------------------
# Environment
# -------------------------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "")  # e.g. https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync

# Keep RAM low on Hobbyist
SESSION = new_session(model_name="u2netp")

app = FastAPI(title="Mockup API (Mask + Outpaint)", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Utilities
# -------------------------
def ensure_env():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
        raise HTTPException(status_code=500, detail="RunPod environment not configured")

def pil_to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def downscale(img: Image.Image, max_side: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

async def read_upload_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    return img

def make_background_alpha_mask(original_rgba: Image.Image) -> Image.Image:
    """
    rembg produces a FOREGROUND mask (white = subject). Our Comfy rule needs
    a BACKGROUND mask in the alpha channel (white = PAINT background, black = KEEP subject).
    So we: downscale -> rembg only_mask -> invert -> gentle feather -> return RGBA with alpha.
    """
    # Work on smaller copy to reduce RAM, then scale mask back up
    small = downscale(original_rgba, max_side=1280)
    buf = io.BytesIO()
    small.save(buf, format="PNG")

    fg_bytes = remove(
        buf.getvalue(),
        session=SESSION,
        only_mask=True,          # 8-bit mask
        post_process_mask=True,  # cleaner edges
    )
    fg_mask_l = Image.open(io.BytesIO(fg_bytes)).convert("L")  # white = subject

    # Invert -> white = background (PAINT), black = subject (KEEP)
    bg_mask_l = ImageOps.invert(fg_mask_l)

    # Light feather to avoid seams at the inpaint boundary
    bg_mask_l = bg_mask_l.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Resize back to original size if we downscaled
    if small.size != original_rgba.size:
        bg_mask_l = bg_mask_l.resize(original_rgba.size, Image.LANCZOS)

    # Place into alpha channel as required by LoadImageMask(channel="alpha")
    mask_rgba = Image.new("RGBA", original_rgba.size, (0, 0, 0, 0))
    mask_rgba.putalpha(bg_mask_l)

    # free
    del small, buf, fg_mask_l, bg_mask_l
    gc.collect()

    return mask_rgba

def html_gallery(items: List[Tuple[str, int]]) -> str:
    imgs = "\n".join(
        f"<figure style='margin:8px'><img style='width:300px;height:auto;display:block' src='data:image/png;base64,{b64}'/>"
        f"<figcaption style='font:12px system-ui;color:#444'>seed {seed}</figcaption></figure>"
        for b64, seed in items
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Generated mockups</title>
</head>
<body style="margin:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto">
  <h1 style="margin:0 0 12px">Generated mockups</h1>
  <p style="margin:0 0 16px;color:#444">Quick preview</p>
  <div style="display:flex;flex-wrap:wrap">{imgs}</div>
</body>
</html>"""

def build_workflow(seed: int, prompt_text: str, negative_text: str = "blurry, artifacts, low quality, watermark, text") -> dict:
    """
    Your provided ComfyUI graph, parameterised for prompt and seed.
    Note: we keep 'image' fields as filenames because we send an 'images' array alongside.
    """
    return {
        # 3: Checkpoint/CLIP/VAE bundle
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": { "ckpt_name": "flux1-dev-fp8.safetensors" }
        },
        # Positive / negative prompts encoded with CLIP from node 3
        "10": {
            "class_type": "CLIPTextEncode",
            "inputs": { "clip": ["3", 1], "text": prompt_text }
        },
        "11": {
            "class_type": "CLIPTextEncode",
            "inputs": { "clip": ["3", 1], "text": negative_text }
        },
        # Load image and mask by name; we'll attach actual base64 images in payload
        "1": { "class_type": "LoadImage", "inputs": { "image": "art.png" } },
        "2": { "class_type": "LoadImageMask", "inputs": { "image": "mask.png", "channel": "alpha" } },
        # Encode for inpaint using mask (white=paint)
        "5": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": { "pixels": ["1", 0], "mask": ["2", 0], "vae": ["3", 2], "grow_mask_by": 24 }
        },
        # Sampler with our seed
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 18,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            }
        },
        # Decode and save
        "7": { "class_type": "VAEDecode", "inputs": { "samples": ["6", 0], "vae": ["3", 2] } },
        "8": { "class_type": "SaveImage", "inputs": { "images": ["7", 0], "filename_prefix": "mockup_out" } }
    }

def call_runpod_with_images(workflow: dict, art_b64: str, mask_b64: str, timeout: int = 120) -> List[str]:
    """
    Sends your workflow plus named images array to RunPod ComfyUI runsync endpoint.
    """
    ensure_env()

    payload = {
        "input": {
            "return_type": "base64",
            "workflow": workflow,
            "images": [
                {"name": "art.png",  "image": art_b64},
                {"name": "mask.png", "image": mask_b64}
            ]
        }
    }
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    try:
        resp = requests.post(RUNPOD_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error calling RunPod: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"RunPod error: {resp.text}")

    data = resp.json()
    images = data.get("output", {}).get("images", [])
    if not images:
        images = [{"image": im} for im in data.get("output", [])]

    b64_list = [im.get("image") for im in images if isinstance(im, dict) and "image" in im]
    if not b64_list:
        raise HTTPException(status_code=502, detail="RunPod returned no images")

    return b64_list


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    # Simple form for manual tests
    return """
<!doctype html>
<html>
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
<body style="margin:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto">
  <h1>Mask + Outpaint</h1>
  <form action="/batch/html" method="post" enctype="multipart/form-data" style="display:grid;gap:8px;max-width:520px">
    <input type="file" name="file" accept="image/*" required />
    <input type="text" name="prompt" placeholder="Describe the room/background…" required />
    <input type="text" name="negative" placeholder="Optional negative prompt" />
    <input type="number" name="count" min="1" max="5" value="1" />
    <button type="submit">Generate</button>
  </form>
  <p style="color:#666">Programmatic endpoint: <code>/batch/json</code></p>
</body>
</html>
""".strip()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/batch/json")
async def batch_json(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative: str = Form("blurry, artifacts, low quality, watermark, text"),
    count: int = Form(1)
):
    """
    Generate 1 image by default. Increase 'count' up to 5 when stable.
    """
    count = max(1, min(5, int(count)))

    # Read input image
    original = await read_upload_image(file)  # RGBA expected

    # Build background mask in alpha channel (white=paint, black=keep)
    mask_rgba = make_background_alpha_mask(original)

    # Encode for RunPod images array
    art_b64 = pil_to_png_b64(original)
    mask_b64 = pil_to_png_b64(mask_rgba)

    # Free heavy objects as soon as possible
    del mask_rgba
    gc.collect()

    results = []
    for _ in range(count):
        seed = secrets.randbits(32)
        wf = build_workflow(seed=seed, prompt_text=prompt, negative_text=negative)
        imgs = call_runpod_with_images(wf, art_b64=art_b64, mask_b64=mask_b64)
        results.append({"seed": seed, "image_b64": imgs[0]})

    # Clean up
    del original, art_b64, mask_b64
    gc.collect()

    return JSONResponse({"count": len(results), "results": results})

@app.post("/batch/html", response_class=HTMLResponse)
async def batch_html(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative: str = Form("blurry, artifacts, low quality, watermark, text"),
    count: int = Form(1)
):
    data = await batch_json(file=file, prompt=prompt, negative=negative, count=count)
    payload = json.loads(data.body.decode("utf-8"))
    items = [(r["image_b64"], r["seed"]) for r in payload["results"]]
    return HTMLResponse(content=html_gallery(items), status_code=200)


# Local dev
if __name__ == "__main__":
    import uvicorn
    # One worker on Hobbyist; no reload in prod
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
