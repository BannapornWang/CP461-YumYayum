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
import torch

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
    pass

app = FastAPI(title="YumYayum Super-Resolution Live Benchmark Studio", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Checkpoint paths
TEAM_52K_PATH = os.path.join(BASE_DIR, "net_g_52000.pth")
OFFICIAL_REALESRGAN_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
ESRGAN_PATH = os.path.join(WEIGHTS_DIR, "ESRGAN_x4.pth")
BSRGAN_PATH = os.path.join(WEIGHTS_DIR, "BSRGAN.pth")
REALESRNET_PATH = os.path.join(WEIGHTS_DIR, "RealESRNet_x4plus.pth")

# Global caches
MODELS: Dict[str, Any] = {}
IQA_METRICS: Dict[str, Any] = {}

def get_iqa_metrics():
    global IQA_METRICS
    if not IQA_METRICS:
        try:
            import pyiqa
            logger.info("Initializing pyiqa metrics (NIQE, MUSIQ)...")
            IQA_METRICS['niqe'] = pyiqa.create_metric('niqe', device=DEVICE)
            IQA_METRICS['musiq'] = pyiqa.create_metric('musiq', device=DEVICE)
        except Exception as e:
            logger.warning(f"Could not load pyiqa: {e}")
    return IQA_METRICS

def convert_bsrgan_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('RRDB_trunk.', 'body.').replace('.RDB1.', '.rdb1.').replace('.RDB2.', '.rdb2.').replace('.RDB3.', '.rdb3.').replace('trunk_conv.', 'conv_body.').replace('HRconv.', 'conv_hr.')
        new_state_dict[new_k] = v
    return new_state_dict

def load_rrdb_model(model_key: str):
    """Direct fast RRDBNet loader for all models"""
    if model_key in MODELS:
        return MODELS[model_key]

    from basicsr.archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    path = None
    is_bsrgan = False

    if model_key == "team_52k" and os.path.exists(TEAM_52K_PATH):
        path = TEAM_52K_PATH
    elif model_key == "realesrgan_official" and os.path.exists(OFFICIAL_REALESRGAN_PATH):
        path = OFFICIAL_REALESRGAN_PATH
    elif model_key == "esrgan" and os.path.exists(ESRGAN_PATH):
        path = ESRGAN_PATH
    elif model_key == "bsrgan" and os.path.exists(BSRGAN_PATH):
        path = BSRGAN_PATH
        is_bsrgan = True
    elif model_key == "realesrnet" and os.path.exists(REALESRNET_PATH):
        path = REALESRNET_PATH

    if not path:
        return None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if is_bsrgan:
        ckpt = convert_bsrgan_keys(ckpt)
        model.load_state_dict(ckpt, strict=False)
    else:
        key = 'params_ema' if 'params_ema' in ckpt else ('params' if 'params' in ckpt else None)
        if key:
            model.load_state_dict(ckpt[key], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
            
    model.eval()
    model = model.to(DEVICE)
    MODELS[model_key] = model
    return model

def image_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, 88] if ext in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(ext, img_bgr, params)
    if not success:
        raise ValueError("Failed to encode image")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"

@app.get("/api/status")
def api_status():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    return {
        "status": "online",
        "device": {"cuda_available": cuda_avail, "gpu_name": gpu_name},
        "available_models": {
            "Bicubic": True,
            "ESRGAN": os.path.exists(ESRGAN_PATH),
            "BSRGAN": os.path.exists(BSRGAN_PATH),
            "Real-ESRNet": os.path.exists(REALESRNET_PATH),
            "Ours (Team 52k)": os.path.exists(TEAM_52K_PATH),
            "Real-ESRGAN (Official)": os.path.exists(OFFICIAL_REALESRGAN_PATH)
        }
    }

@app.post("/api/compare")
async def compare_all_models(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    outscale: float = Form(4.0)
):
    """Run all models simultaneously and compute NIQE / MUSIQ metrics"""
    img_bgr = None
    if file is not None and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif preset:
        # Check presets in datasets/eval_20/LR or inputs
        p1 = os.path.join(BASE_DIR, "datasets", "eval_20", "LR", preset)
        p2 = os.path.join(INPUTS_DIR, preset)
        target_p = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        if target_p:
            img_bgr = cv2.imread(target_p, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No valid image provided")

    h, w = img_bgr.shape[:2]
    # Resize input if very large to prevent memory overhead during multi-model demo
    max_side = 512
    if max(h, w) > max_side:
        scale_ratio = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale_ratio), int(h * scale_ratio)), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    input_b64 = image_to_base64(img_bgr, ext=".jpg")
    iqa = get_iqa_metrics()

    model_registry = [
        ("Bicubic (Baseline)", "bicubic", None),
        ("ESRGAN", "esrgan", ESRGAN_PATH),
        ("BSRGAN", "bsrgan", BSRGAN_PATH),
        ("Real-ESRNet", "realesrnet", REALESRNET_PATH),
        ("Real-ESRGAN (โมเดลของทีม - 52k)", "team_52k", TEAM_52K_PATH),
        ("Real-ESRGAN (Official)", "realesrgan_official", OFFICIAL_REALESRGAN_PATH)
    ]

    results = []

    # Tensor preparation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    for display_name, m_key, m_path in model_registry:
        t0 = time.perf_counter()
        out_bgr = None
        
        try:
            if m_key == "bicubic":
                out_bgr = cv2.resize(img_bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
            else:
                if not m_path or not os.path.exists(m_path):
                    continue
                net = load_rrdb_model(m_key)
                if net is None:
                    continue
                with torch.no_grad():
                    sr_tensor = net(tensor)
                    sr_np = sr_tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
                    out_bgr = (sr_np.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')

            t_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Compute IQA metrics
            niqe_score, musiq_score = 0.0, 0.0
            if iqa and out_bgr is not None:
                tmp_path = os.path.join(BASE_DIR, f"_tmp_eval_{m_key}.jpg")
                cv2.imwrite(tmp_path, out_bgr)
                with torch.no_grad():
                    if 'niqe' in iqa:
                        niqe_score = round(iqa['niqe'](tmp_path).item(), 3)
                    if 'musiq' in iqa:
                        musiq_score = round(iqa['musiq'](tmp_path).item(), 2)
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            results.append({
                "model": display_name,
                "is_team": "ทีม" in display_name or "team" in display_name.lower(),
                "latency_ms": t_ms,
                "output_shape": f"{w*4}x{h*4}",
                "niqe": niqe_score,
                "musiq": musiq_score,
                "image": image_to_base64(out_bgr, ext=".jpg")
            })

        except Exception as e:
            logger.error(f"Error evaluating {display_name}: {e}")

    return {
        "success": True,
        "input_dimensions": {"width": w, "height": h},
        "input_image": input_b64,
        "results": results
    }

# Serve static frontend & reports
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
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=8000, reload=False)
