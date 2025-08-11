import io
import os
import json
import base64
import secrets
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

app = FastAPI(
    title="Mockup API",
    description="Upload an image and generate wall-art mockups via RunPod (mask worker + ComfyUI).",
    version="2.0"
)

# ------- env -------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
MASK_ENDPOINT = os.getenv("RUNPOD_MASK_ENDPOINT")       # e.g. https://api.runpod.ai/v2/<MASK_ID>/runsync
COMFY_ENDPOINT = os.getenv("RUNPOD_COMFY_ENDPOINT")     # e.g. https://api.runpod.ai/v2/<COMFY_ID>/runsync

def ensure_env():
    missing = []
    if not RUNPOD_API_KEY: missing.append("RUNPOD_API_KEY")
    if not MASK_ENDPOINT: missing.append("RUNPOD_MASK_ENDPOINT")
    if not COMFY_ENDPOINT: missing.append("RUNPOD_COMFY_ENDPOINT")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env: {', '.join(missing)}")

# ------- style presets for digital wall art -------
STYLE_PRESETS: Dict[str, str] = {
    "modern_living_room": "A realistic interior photo of a modern living room with a large framed art piece on a light grey wall above a mid-century style sofa. Soft natural daylight from the side, gentle shadows, harmonious neutral tones, subtle wall texture. The artwork is perfectly aligned, scale and perspective match the room with depth and realism.",
    "minimal_gallery": "A bright minimalist gallery space with white walls and light oak flooring featuring a single large framed art piece as the focal point. Even diffused lighting clean shadows balanced perspective and soft depth in the background. Colours are harmonious giving the artwork a premium exhibition feel.",
    "reading_nook": "A cosy reading nook with a framed art piece on the wall above a comfortable armchair soft ambient lighting from a nearby lamp and warm daylight from a window. Subtle depth realistic shadows and harmonious tones that make the artwork feel naturally part of the space.",
    "scandi_bedroom": "A Scandinavian style bedroom with a framed art piece above the bed light pastel wall colours linen bedding and natural wood furniture. Bright diffused daylight from the window realistic soft shadows and perfect alignment of the art to match scale and perspective.",
    "industrial_loft": "An industrial style loft with exposed brick walls metal window frames and a framed art piece mounted centrally. Natural daylight streaming through tall windows realistic depth shadows matching the light direction and warm yet balanced tones.",
    "elegant_hallway": "A bright elegant hallway with a single framed art piece hanging on a cream coloured wall. Subtle depth with the hallway receding in the background soft natural light from side windows and balanced colours that highlight the artwork without overpowering it."
}

def resolve_prompt(style: str, prompt: str) -> str:
    return STYLE_PRESETS.get(style) if style else (prompt or "A bright minimalist gallery space with soft daylight")

# ------- helpers -------
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def call_runpod(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"RunPod error: {r.text}")
    return r.json()

def call_mask_worker(img_b64: str, mode: str = "preview") -> str:
    # mode: "preview" or "final"
    payload = {"input": {"image_b64": img_b64, "mode": mode}}
    data = call_runpod(MASK_ENDPOINT, payload)
    # Some endpoints wrap in { output: { mask_b64 } }, others return { mask_b64 }
    return (data.get("output") or {}).get("mask_b64") or data.get("mask_b64")

def comfy_workflow(seed: int, prompt: str) -> Dict[str, Any]:
    # Uses CLIPTextEncode nodes and applies mask to latent before sampling.
    return {
        "img":   {"class_type": "LoadImage",      "inputs": {"image": "__b64_img__"}},
        "mask":  {"class_type": "LoadImageMask",  "inputs": {"image": "__b64_mask__", "channel": "alpha"}},
        "pos":   {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
        "neg":   {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "prep":  {"class_type": "ApplyMaskToLatent", "inputs": {"image": ["img", 0], "mask": ["mask", 0]}},
        "noise": {"class_type": "RandomNoise",    "inputs": {"seed": seed}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["checkpoint", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["prep", 0],
                "noise": ["noise", 0],
                "steps": 22,
                "cfg": 7.5,
                "sampler_name": "euler"
            }
        },
        "out": {"class_type": "SaveImage", "inputs": {"images": ["sampler", 0]}}
    }

def call_comfy(img_b64: str, mask_b64: str, prompt: str, seed: int) -> List[str]:
    wf = json.loads(
        json.dumps(comfy_workflow(seed, prompt))
        .replace("__b64_img__", img_b64)
        .replace("__b64_mask__", mask_b64)
    )
    payload = {"input": {"return_type": "base64", "workflow": wf}}
    data = call_runpod(COMFY_ENDPOINT, payload)
    images = data.get("output", {}).get("images", [])
    if not images:
        images = [{"image": x} for x in data.get("output", [])]
    return [e["image"] for e in images if isinstance(e, dict) and "image" in e]

# ------- endpoints -------
@app.get("/healthz")
def healthz():
    try:
        ensure_env()
        return {"ok": True}
    except HTTPException as e:
        return {"ok": False, "detail": e.detail}

@app.post("/mask/test", summary="Test mask worker (returns mask only)")
async def mask_test(
    file: UploadFile = File(...),
    mode: str = Form("preview")  # "preview" or "final"
):
    ensure_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    img_b64 = pil_to_b64(img)
    mask_b64 = call_mask_worker(img_b64, mode=mode)
    return {"mode": mode, "mask_b64": mask_b64}

@app.post("/preview/json", summary="Preview mockups (JSON, 5 variations)")
async def preview_json(
    file: UploadFile = File(...),
    style: str = Form(""),
    prompt: str = Form(""),
    mode: str = Form("preview")  # pass "final" if you want best mask in preview
):
    ensure_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    img_b64 = pil_to_b64(img)
    use_prompt = resolve_prompt(style, prompt)
    mask_b64 = call_mask_worker(img_b64, mode=mode)

    seeds = [secrets.randbits(32) for _ in range(5)]
    results = []
    for s in seeds:
        imgs = call_comfy(img_b64, mask_b64, use_prompt, seed=s)
        if imgs:
            results.append({"seed": s, "image_b64": imgs[0]})

    return {
        "mode": mode,
        "style": style or "custom",
        "prompt": use_prompt,
        "count": len(results),
        "results": results
    }

@app.post("/preview/html", summary="Preview mockups (HTML gallery, 5 variations)", response_class=HTMLResponse)
async def preview_html(
    file: UploadFile = File(...),
    style: str = Form(""),
    prompt: str = Form(""),
    mode: str = Form("preview")
):
    ensure_env()
    img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    img_b64 = pil_to_b64(img)
    use_prompt = resolve_prompt(style, prompt)
    mask_b64 = call_mask_worker(img_b64, mode=mode)

    seeds = [secrets.randbits(32) for _ in range(5)]
    imgs64: List[str] = []
    for s in seeds:
        out = call_comfy(img_b64, mask_b64, use_prompt, seed=s)
        if out:
            imgs64.append(out[0])

    html = "<h1>Generated Mockups</h1><div style='display:flex;flex-wrap:wrap'>"
    for img64 in imgs64:
        html += f"<img src='data:image/png;base64,{img64}' style='width:300px;margin:5px'/>"
    html += "</div>"
    return HTMLResponse(content=html)
