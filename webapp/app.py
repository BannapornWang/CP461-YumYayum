import os
import io
import sys
import time
import base64
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YumYayumApp")

# Ensure root dir is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Monkey-patch for BasicSR on newer torchvision
try:
    from torchvision.transforms import functional as torchvision_functional
    sys.modules.setdefault('torchvision.transforms.functional_tensor', torchvision_functional)
except ImportError:
    logger.warning("torchvision not fully initialized yet; will retry on model load.")

app = FastAPI(title="YumYayum Super-Resolution Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
TEAM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "train_RealESRNetx4_CP461_team_5h",
    "models",
    "net_g_9000.pth"
)
OFFICIAL_MODEL_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

# Global model cache
MODELS: Dict[str, Any] = {}

def get_device_info():
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        vram_total = 0.0
        vram_alloc = 0.0
        if cuda_avail:
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            vram_alloc = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        return {
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total,
            "vram_allocated_gb": vram_alloc
        }
    except Exception as e:
        return {
            "cuda_available": False,
            "gpu_name": "CPU (Torch not loaded)",
            "vram_total_gb": 0.0,
            "vram_allocated_gb": 0.0,
            "error": str(e)
        }

def load_sr_model(model_key: str, tile: int = 256):
    """Lazy load Real-ESRGAN / Real-ESRNet models"""
    cache_id = f"{model_key}_tile_{tile}"
    if cache_id in MODELS:
        return MODELS[cache_id]

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url

    device_id = 0 if torch.cuda.is_available() else None
    half_precision = torch.cuda.is_available()

    if model_key == "team_9000":
        if not os.path.exists(TEAM_MODEL_PATH):
            raise FileNotFoundError(f"Checkpoint not found at: {TEAM_MODEL_PATH}")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=TEAM_MODEL_PATH,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half_precision,
            gpu_id=device_id
        )
        MODELS[cache_id] = upsampler
        return upsampler

    elif model_key == "realesrgan_x4plus":
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        model_path = OFFICIAL_MODEL_PATH
        if not os.path.exists(model_path):
            logger.info("Downloading official RealESRGAN_x4plus.pth...")
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            model_path = load_file_from_url(url=url, model_dir=WEIGHTS_DIR, progress=True)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half_precision,
            gpu_id=device_id
        )
        MODELS[cache_id] = upsampler
        return upsampler
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

def image_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(ext, img_bgr, params)
    if not success:
        raise ValueError("Failed to encode image")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"

@app.get("/api/status")
def api_status():
    dev = get_device_info()
    return {
        "status": "online",
        "device": dev,
        "team_checkpoint_exists": os.path.exists(TEAM_MODEL_PATH),
        "team_checkpoint_size_mb": round(os.path.getsize(TEAM_MODEL_PATH)/(1024*1024), 2) if os.path.exists(TEAM_MODEL_PATH) else 0,
        "official_weights_exist": os.path.exists(OFFICIAL_MODEL_PATH),
        "supported_models": [
            {"id": "team_9000", "name": "CP461 Team Real-ESRNet (9k iters)", "desc": "Trained by Team YumYayum (5-hour run checkpoint)"},
            {"id": "realesrgan_x4plus", "name": "Real-ESRGAN x4plus (Official)", "desc": "Full official pre-trained GAN weights"},
            {"id": "bicubic", "name": "Bicubic Interpolation 4x", "desc": "Standard mathematical baseline (No Neural Net)"}
        ],
        "reports": [
            {"id": "benchmark", "name": "Benchmark Results (20 Images)", "url": "/benchmark"},
            {"id": "eda", "name": "EDA & Data Exploration", "url": "/eda"},
            {"id": "comparison_table", "name": "20-Image Comparison Table", "url": "/comparison-table"},
            {"id": "gallery", "name": "Model Visual Gallery", "url": "/gallery"}
        ]
    }

@app.get("/api/presets")
def api_presets():
    presets = []
    if os.path.isdir(INPUTS_DIR):
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        for f in sorted(os.listdir(INPUTS_DIR)):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                presets.append({
                    "filename": f,
                    "title": f.replace("_", " ").title(),
                    "url": f"/api/inputs/{f}"
                })
    return {"presets": presets}

@app.get("/api/inputs/{filename}")
def get_input_file(filename: str):
    path = os.path.join(INPUTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Preset image not found")
    return FileResponse(path)

@app.post("/api/enhance")
async def enhance_image(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    model: str = Form("team_9000"),
    tile: int = Form(256),
    outscale: float = Form(4.0)
):
    # 1. Read input image
    img_bgr = None
    original_name = "image"
    if file is not None and file.filename:
        original_name = file.filename
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    elif preset:
        original_name = preset
        preset_path = os.path.join(INPUTS_DIR, preset)
        if os.path.isfile(preset_path):
            img_bgr = cv2.imread(preset_path, cv2.IMREAD_UNCHANGED)
        else:
            raise HTTPException(status_code=400, detail=f"Preset {preset} not found")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No valid image provided")

    h, w = img_bgr.shape[:2]
    start_time = time.perf_counter()

    # 2. Super-resolution processing
    enhanced_bgr = None
    has_alpha = len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4

    try:
        if model == "bicubic":
            new_w, new_h = int(w * outscale), int(h * outscale)
            enhanced_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            upsampler = load_sr_model(model, tile=tile)
            enhanced_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale)
    except Exception as e:
        logger.error(f"Inference error with model {model}: {e}", exc_info=True)
        # Fallback to bicubic if GPU OOM or missing package
        new_w, new_h = int(w * outscale), int(h * outscale)
        enhanced_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "fallback_used": True
        })

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    out_h, out_w = enhanced_bgr.shape[:2]

    # Convert to base64 Data URLs
    ext = ".png" if has_alpha else ".jpg"
    input_b64 = image_to_base64(img_bgr, ext=ext)
    output_b64 = image_to_base64(enhanced_bgr, ext=ext)

    return {
        "success": True,
        "model_used": model,
        "input_dimensions": {"width": w, "height": h},
        "output_dimensions": {"width": out_w, "height": out_h},
        "scale": outscale,
        "latency_ms": elapsed_ms,
        "input_image": input_b64,
        "output_image": output_b64,
        "filename": original_name
    }

@app.post("/api/compare")
async def compare_all_models(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    tile: int = Form(256),
    outscale: float = Form(4.0)
):
    """Run Bicubic, Team Real-ESRNet, and Real-ESRGAN in one request for benchmark comparison"""
    img_bgr = None
    if file is not None and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    elif preset:
        preset_path = os.path.join(INPUTS_DIR, preset)
        if os.path.isfile(preset_path):
            img_bgr = cv2.imread(preset_path, cv2.IMREAD_UNCHANGED)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No image provided")

    h, w = img_bgr.shape[:2]
    # Use PNG only if image has alpha channel (4 channels); otherwise JPEG
    has_alpha = img_bgr.ndim == 3 and img_bgr.shape[2] == 4
    ext = ".png" if has_alpha else ".jpg"
    input_b64 = image_to_base64(img_bgr, ext=ext)

    results = {}
    models_to_test = ["bicubic", "team_9000"]
    if os.path.exists(OFFICIAL_MODEL_PATH):
        models_to_test.append("realesrgan_x4plus")

    for m in models_to_test:
        t0 = time.perf_counter()
        try:
            if m == "bicubic":
                out = cv2.resize(img_bgr, (int(w * outscale), int(h * outscale)), interpolation=cv2.INTER_CUBIC)
            else:
                upsampler = load_sr_model(m, tile=tile)
                out, _ = upsampler.enhance(img_bgr, outscale=outscale)
            t_ms = round((time.perf_counter() - t0) * 1000, 1)
            results[m] = {
                "success": True,
                "latency_ms": t_ms,
                "image": image_to_base64(out, ext=ext)
            }
        except Exception as e:
            results[m] = {
                "success": False,
                "error": str(e)
            }

    return {
        "input_dimensions": {"width": w, "height": h},
        "input_image": input_b64,
        "results": results
    }

# Serve static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to CP461 YumYayum Super-Resolution API"}

@app.get("/benchmark")
def serve_benchmark():
    file_path = os.path.join(STATIC_DIR, "benchmark_results.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Benchmark results not found")

@app.get("/eda")
def serve_eda():
    file_path = os.path.join(STATIC_DIR, "eda_dashboard.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="EDA dashboard not found")

@app.get("/comparison-table")
def serve_comparison_table():
    file_path = os.path.join(STATIC_DIR, "model_comparison_20_table.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Comparison table not found")

@app.get("/gallery")
def serve_gallery():
    file_path = os.path.join(STATIC_DIR, "model_comparison_gallery.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Model comparison gallery not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=True)
