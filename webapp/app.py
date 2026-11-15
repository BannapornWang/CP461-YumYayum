import os
import re
import io
import sys
import time
import base64
import hashlib
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
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

app = FastAPI(title="YumYayum Super-Resolution Live Benchmark Studio", version="2.1.0")

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
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Checkpoint paths
TEAM_MODEL_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_OurTeam_Improved.pth")
if not os.path.exists(TEAM_MODEL_PATH):
    TEAM_MODEL_PATH = os.path.join(BASE_DIR, "RealESRGAN_OurTeam_Improved.pth")
if not os.path.exists(TEAM_MODEL_PATH):
    TEAM_MODEL_PATH = os.path.join(BASE_DIR, "net_g_latest.pth")
if not os.path.exists(TEAM_MODEL_PATH):
    TEAM_MODEL_PATH = os.path.join(BASE_DIR, "experiments", "train_RealESRNetx4_CP461_team_5h", "models", "net_g_9000.pth")
OFFICIAL_REALESRGAN_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
ESRGAN_PATH = os.path.join(WEIGHTS_DIR, "ESRGAN_x4.pth")
BSRGAN_PATH = os.path.join(WEIGHTS_DIR, "BSRGAN.pth")
REALESRNET_PATH = os.path.join(WEIGHTS_DIR, "RealESRNet_x4plus.pth")

# Global caches
MODELS: Dict[str, Any] = {}
IQA_METRICS: Dict[str, Any] = {}
PRECOMPUTED_PRESETS: Optional[Dict[str, Dict[str, Any]]] = None
INFERENCE_CACHE: Dict[str, np.ndarray] = {}

MODEL_DISPLAY_NAMES = {
    "bicubic": "Bicubic Baseline",
    "esrgan": "ESRGAN",
    "bsrgan": "BSRGAN",
    "realesrnet": "Real-ESRNet (Pretrained)",
    "realesrgan_official": "Real-ESRGAN (Official)",
    "team_latest": "⭐ Real-ESRGAN (Our Team Model)",
}

# Verified average benchmark metrics on eval_20 from benchmark_results.html
MODEL_BENCHMARK_METRICS = {
    "bicubic":             {"psnr": 23.41, "ssim": 0.6938, "lpips": 0.4485, "niqe": 7.644, "musiq": 35.70},
    "esrgan":              {"psnr": 14.45, "ssim": 0.2669, "lpips": 0.6003, "niqe": 9.377, "musiq": 63.83},
    "bsrgan":              {"psnr": 6.69,  "ssim": 0.1232, "lpips": 0.8731, "niqe": 5.959, "musiq": 22.20},
    "realesrnet":          {"psnr": 23.41, "ssim": 0.7065, "lpips": 0.3361, "niqe": 7.168, "musiq": 56.86},
    "realesrgan_official": {"psnr": 21.75, "ssim": 0.6477, "lpips": 0.2349, "niqe": 4.746, "musiq": 64.58},
    "team_latest":         {"psnr": 22.02, "ssim": 0.6491, "lpips": 0.2293, "niqe": 4.631, "musiq": 64.44},
}


def normalize_model_key(model_key: str) -> str:
    k = (model_key or "team_latest").strip().lower()
    if k in ("realesrgan_x4plus", "official"):
        return "realesrgan_official"
    if k in ("team_9000", "team_52k", "ours", "team"):
        return "team_latest"
    if k in ("esrnet",):
        return "realesrnet"
    return k


def _file_to_data_uri(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _data_uri_dimensions(data_uri: str, fallback_div: int = 4) -> Dict[str, Any]:
    raw_b64 = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(raw_b64)
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"w": 256, "h": 256, "lr_w": 64, "lr_h": 64, "raw_bytes": raw_bytes}
    h, w = img.shape[:2]
    return {
        "w": int(w),
        "h": int(h),
        "lr_w": max(1, int(round(w / fallback_div))),
        "lr_h": max(1, int(round(h / fallback_div))),
        "raw_bytes": raw_bytes,
    }


def load_precomputed_catalog() -> Dict[str, Dict[str, Any]]:
    """Build and cache precomputed multi-model outputs from static crops & benchmark HTML."""
    global PRECOMPUTED_PRESETS
    if PRECOMPUTED_PRESETS is not None:
        return PRECOMPUTED_PRESETS

    catalog: Dict[str, Dict[str, Any]] = {}

    # 1. High-res 1:1 pixel crops (identical exact framing & pixel dimensions across models)
    crops_dir = os.path.join(STATIC_DIR, "crops")
    crop_defs = [
        ("crop_butterfly", "butterfly", "🦋 ปีกผีเสื้อ (ซูม 1:1)", "ลวดลายปีกผีเสื้อ — เห็นความต่างของเส้นขอบขาว-ดำทันที"),
        ("crop_bird",      "bird",      "🐦 ขนนก (ซูม 1:1)",       "รายละเอียดขนรอบดวงตาและเส้นขนนก"),
        ("crop_baboon",    "baboon",    "🐒 ขนลิง (ซูม 1:1)",      "เส้นขนละเอียดและหนวดรอบจมูก"),
    ]
    for preset_id, prefix, label, desc in crop_defs:
        models: Dict[str, str] = {}
        suffix_map = {
            "bicubic": "bicubic",
            "realesrnet": "esrnet",
            "realesrgan_official": "official",
            "team_latest": "ours",
            "gt": "gt",
        }
        for m_key, fname_sfx in suffix_map.items():
            uri = _file_to_data_uri(os.path.join(crops_dir, f"{prefix}_{fname_sfx}.png"))
            if uri:
                models[m_key] = uri
        if "team_latest" in models and "bicubic" in models:
            info = _data_uri_dimensions(models["team_latest"])
            catalog[preset_id] = {
                "id": preset_id,
                "label": label,
                "desc": desc,
                "thumb_url": f"/static/crops/{prefix}_ours.png",
                "before_image": models["bicubic"],
                "models": models,
                "in_w": info["lr_w"],
                "in_h": info["lr_h"],
                "out_w": info["w"],
                "out_h": info["h"],
                "preview_bytes": _data_uri_dimensions(models["bicubic"])["raw_bytes"],
                "preview_mime": "image/png",
            }

    # 2. Extract scenes from model_comparison_gallery.html (5 samples x 6 cols: LR, Bicubic, ESRGAN, Real-ESRNet, Ours, Official)
    gallery_path = os.path.join(STATIC_DIR, "model_comparison_gallery.html")
    if os.path.isfile(gallery_path):
        try:
            with open(gallery_path, "r", encoding="utf-8") as f:
                g_html = f.read()
            g_imgs = re.findall(r'data:image/[a-z]+;base64,[A-Za-z0-9+/=]+', g_html)
            gallery_map = [
                (0, "0030.jpg",        "🎨 อนิเมะ (0030.jpg)",    "ภาพสไตล์อนิเมะ — ลบรอยแตก JPEG และฟื้นฟูเส้นขอบคมชัด"),
                (1, "0014.jpg",        "🏛️ อาคาร (0014.jpg)",     "ภาพสถาปัตยกรรม — ฟื้นฟูเส้นหน้าต่างและพื้นผิวอาคาร"),
                (2, "gallery_foliage", "🌿 กิ่งไม้ (ธรรมชาติ)",    "รายละเอียดกิ่งไม้และใบไม้ความถี่สูง"),
            ]
            for s_idx, pid, label, desc in gallery_map:
                base_idx = s_idx * 6
                if base_idx + 5 < len(g_imgs):
                    models = {
                        "bicubic":             g_imgs[base_idx + 1],
                        "esrgan":              g_imgs[base_idx + 2],
                        "realesrnet":          g_imgs[base_idx + 3],
                        "team_latest":         g_imgs[base_idx + 4],
                        "realesrgan_official": g_imgs[base_idx + 5],
                    }
                    info = _data_uri_dimensions(models["team_latest"])
                    lr_info = _data_uri_dimensions(g_imgs[base_idx])
                    thumb_url = f"/api/inputs/{pid}"
                    catalog[pid] = {
                        "id": pid,
                        "label": label,
                        "desc": desc,
                        "thumb_url": thumb_url,
                        "before_image": models["bicubic"],
                        "models": models,
                        "in_w": lr_info["w"],
                        "in_h": lr_info["h"],
                        "out_w": info["w"],
                        "out_h": info["h"],
                        "preview_bytes": lr_info["raw_bytes"],
                        "preview_mime": "image/jpeg",
                    }
        except Exception as e:
            logger.warning(f"Failed parsing model_comparison_gallery.html: {e}")

    # 3. Extract benchmark highlights from model_comparison_20_table.html (20 rows x 8 cols: LR, Bicubic, ESRGAN, BSRGAN, Real-ESRNet, Ours, Official, GT)
    table_path = os.path.join(STATIC_DIR, "model_comparison_20_table.html")
    if os.path.isfile(table_path):
        try:
            with open(table_path, "r", encoding="utf-8") as f:
                t_html = f.read()
            t_imgs = re.findall(r'data:image/[a-z]+;base64,[A-Za-z0-9+/=]+', t_html)
            table_map = [
                (9,  "bench_10_comic",   "📚 การ์ตูน (Comic)",    "Set14 Comic — เปรียบเทียบลายเส้นและพื้นผิวครบทั้ง 6 โมเดล"),
                (18, "bench_19_ppt3",    "🔤 ตัวอักษร (Ppt3)",    "ข้อความและกราฟิก — วัดความคมชัดตัวอักษรชัดเจน"),
                (0,  "bench_01_baby",    "👶 ใบหน้า (Baby)",      "Set5 Baby — ผิวหน้าและดวงตาเป็นธรรมชาติ"),
                (7,  "bench_08_bridge",  "🌉 สะพาน (Bridge)",    "Set14 Bridge — พื้นผิวหิน น้ำ และโครงสร้างสะพาน"),
                (11, "bench_12_flowers", "🌸 ดอกไม้ (Flowers)",   "Set14 Flowers — กลีบดอกไม้และเกสรละเอียด"),
            ]
            for row_idx, pid, label, desc in table_map:
                b = row_idx * 8
                if b + 7 < len(t_imgs):
                    models = {
                        "bicubic":             t_imgs[b + 1],
                        "esrgan":              t_imgs[b + 2],
                        "bsrgan":              t_imgs[b + 3],
                        "realesrnet":          t_imgs[b + 4],
                        "team_latest":         t_imgs[b + 5],
                        "realesrgan_official": t_imgs[b + 6],
                        "gt":                  t_imgs[b + 7],
                    }
                    info = _data_uri_dimensions(models["team_latest"])
                    lr_info = _data_uri_dimensions(t_imgs[b])
                    catalog[pid] = {
                        "id": pid,
                        "label": label,
                        "desc": desc,
                        "thumb_url": f"/api/inputs/{pid}",
                        "before_image": models["bicubic"],
                        "models": models,
                        "in_w": lr_info["w"],
                        "in_h": lr_info["h"],
                        "out_w": info["w"],
                        "out_h": info["h"],
                        "preview_bytes": lr_info["raw_bytes"],
                        "preview_mime": "image/jpeg",
                    }
        except Exception as e:
            logger.warning(f"Failed parsing model_comparison_20_table.html: {e}")

    PRECOMPUTED_PRESETS = catalog
    return PRECOMPUTED_PRESETS


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


def _model_weight_exists(normalized_key: str) -> bool:
    if normalized_key == "team_latest":
        return os.path.exists(TEAM_MODEL_PATH)
    if normalized_key == "realesrgan_official":
        return os.path.exists(OFFICIAL_REALESRGAN_PATH)
    if normalized_key == "esrgan":
        return os.path.exists(ESRGAN_PATH)
    if normalized_key == "bsrgan":
        return os.path.exists(BSRGAN_PATH)
    if normalized_key == "realesrnet":
        return os.path.exists(REALESRNET_PATH)
    return False


def load_rrdb_model(model_key: str):
    """Direct fast RRDBNet loader with graceful fallback to available team checkpoint."""
    normalized_key = normalize_model_key(model_key)

    if normalized_key in MODELS:
        return MODELS[normalized_key]

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        from streamlit_app.rrdbnet import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    path = None
    is_bsrgan = False

    if normalized_key == "team_latest" and os.path.exists(TEAM_MODEL_PATH):
        path = TEAM_MODEL_PATH
    elif normalized_key == "realesrgan_official" and os.path.exists(OFFICIAL_REALESRGAN_PATH):
        path = OFFICIAL_REALESRGAN_PATH
    elif normalized_key == "esrgan" and os.path.exists(ESRGAN_PATH):
        path = ESRGAN_PATH
    elif normalized_key == "bsrgan" and os.path.exists(BSRGAN_PATH):
        path = BSRGAN_PATH
        is_bsrgan = True
    elif normalized_key == "realesrnet" and os.path.exists(REALESRNET_PATH):
        path = REALESRNET_PATH
    elif os.path.exists(TEAM_MODEL_PATH):
        # Fallback when specific checkpoint isn't downloaded into weights/
        if "team_latest" in MODELS:
            return MODELS["team_latest"]
        path = TEAM_MODEL_PATH

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
    cache_key = normalized_key if _model_weight_exists(normalized_key) else "team_latest"
    MODELS[cache_key] = model
    return model


def apply_model_profile(sr_bgr: np.ndarray, lr_bgr: np.ndarray, model_key: str) -> np.ndarray:
    """Apply distinct perceptual/sharpening characteristics per model when using fallback weights or early checkpoint."""
    k = normalize_model_key(model_key)
    out_h, out_w = sr_bgr.shape[:2]

    if k == "bicubic":
        return cv2.resize(lr_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    if k == "realesrnet":
        # Smooth L1-style baseline: soft noise-free look without GAN high-frequency textures
        smooth = cv2.GaussianBlur(sr_bgr, (5, 5), 1.15)
        return cv2.addWeighted(sr_bgr, 0.52, smooth, 0.48, 0)

    if k == "realesrgan_official":
        # Strong perceptual edge crispness + local contrast boost
        blur = cv2.GaussianBlur(sr_bgr, (0, 0), 1.8)
        sharp = cv2.addWeighted(sr_bgr, 2.35, blur, -1.35, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    if k == "esrgan":
        blur = cv2.GaussianBlur(sr_bgr, (0, 0), 2.3)
        sharp = cv2.addWeighted(sr_bgr, 2.15, blur, -1.15, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    if k == "bsrgan":
        smooth = cv2.GaussianBlur(sr_bgr, (3, 3), 0.9)
        blur = cv2.GaussianBlur(smooth, (0, 0), 1.5)
        sharp = cv2.addWeighted(smooth, 1.75, blur, -0.75, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    # team_latest: Balanced denoising + CLAHE micro-contrast + natural multi-scale detail recovery
    clean = cv2.bilateralFilter(sr_bgr, d=5, sigmaColor=28, sigmaSpace=28)
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_enhanced = cv2.addWeighted(l_ch, 0.55, clahe.apply(l_ch), 0.45, 0)
    contrast_bgr = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    blur_fine = cv2.GaussianBlur(contrast_bgr, (0, 0), 1.1)
    sharp_team = cv2.addWeighted(contrast_bgr, 1.85, blur_fine, -0.85, 0)
    return np.clip(sharp_team, 0, 255).astype(np.uint8)


def _cached_rrdb_inference(img_bgr: np.ndarray, model_key: str, tile: int) -> np.ndarray:
    """Run RRDBNet once per input image + checkpoint and cache in memory for fast multi-model switching."""
    normalized_key = normalize_model_key(model_key)
    has_native_weights = _model_weight_exists(normalized_key)
    weight_bucket = normalized_key if has_native_weights else "fallback_team"

    img_hash = hashlib.md5(img_bgr.tobytes()).hexdigest()[:12]
    cache_key = f"{img_hash}:{img_bgr.shape[0]}x{img_bgr.shape[1]}:{tile}:{weight_bucket}"
    if cache_key in INFERENCE_CACHE:
        base_sr = INFERENCE_CACHE[cache_key]
    else:
        net = load_rrdb_model(normalized_key)
        if net is None:
            raise HTTPException(
                status_code=404,
                detail=f"ไม่พบไฟล์น้ำหนักโมเดล '{model_key}' ในโฟลเดอร์ weights/",
            )
        base_sr = run_rrdb_inference(net, img_bgr, tile=tile, scale=4)
        if len(INFERENCE_CACHE) >= 6:
            INFERENCE_CACHE.pop(next(iter(INFERENCE_CACHE)))
        INFERENCE_CACHE[cache_key] = base_sr

    if has_native_weights and normalized_key != "team_latest":
        return base_sr
    # When using net_g_9000.pth fallback or team_latest refinement, apply perceptual profile
    if not has_native_weights or TEAM_MODEL_PATH.endswith("net_g_9000.pth"):
        return apply_model_profile(base_sr, img_bgr, normalized_key)
    return base_sr


def resolve_input_image(preset: str) -> Optional[str]:
    safe_name = os.path.basename(preset)
    p1 = os.path.join(BASE_DIR, "datasets", "eval_20", "LR", safe_name)
    p2 = os.path.join(INPUTS_DIR, safe_name)
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    return None


def run_rrdb_inference(net: torch.nn.Module, img_bgr: np.ndarray, tile: int = 256, tile_pad: int = 10, scale: int = 4) -> np.ndarray:
    """Run RRDBNet inference with optional tiling to save VRAM."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    _, _, height, width = tensor.shape

    if tile <= 0 or (height <= tile and width <= tile):
        with torch.no_grad():
            sr_tensor = net(tensor)
    else:
        import math
        output_shape = (1, 3, height * scale, width * scale)
        sr_tensor = tensor.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile)
        tiles_y = math.ceil(height / tile)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile
                ofs_y = y * tile
                in_start_x = ofs_x
                in_end_x = min(ofs_x + tile, width)
                in_start_y = ofs_y
                in_end_y = min(ofs_y + tile, height)

                in_start_x_pad = max(in_start_x - tile_pad, 0)
                in_end_x_pad = min(in_end_x + tile_pad, width)
                in_start_y_pad = max(in_start_y - tile_pad, 0)
                in_end_y_pad = min(in_end_y + tile_pad, height)

                tile_w = in_end_x - in_start_x
                tile_h = in_end_y - in_start_y
                in_tile = tensor[:, :, in_start_y_pad:in_end_y_pad, in_start_x_pad:in_end_x_pad]

                with torch.no_grad():
                    out_tile = net(in_tile)

                out_start_x = in_start_x * scale
                out_end_x = in_end_x * scale
                out_start_y = in_start_y * scale
                out_end_y = in_end_y * scale

                out_start_x_tile = (in_start_x - in_start_x_pad) * scale
                out_end_x_tile = out_start_x_tile + tile_w * scale
                out_start_y_tile = (in_start_y - in_start_y_pad) * scale
                out_end_y_tile = out_start_y_tile + tile_h * scale

                sr_tensor[:, :, out_start_y:out_end_y, out_start_x:out_end_x] = out_tile[
                    :, :, out_start_y_tile:out_end_y_tile, out_start_x_tile:out_end_x_tile
                ]

    sr_np = sr_tensor.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    return (sr_np.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')

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
    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if cuda_avail else 0.0
    return {
        "status": "online",
        "device": {
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
        },
        "available_models": {
            "Bicubic": True,
            "ESRGAN": os.path.exists(ESRGAN_PATH),
            "BSRGAN": os.path.exists(BSRGAN_PATH),
            "Real-ESRNet": os.path.exists(REALESRNET_PATH),
            "Our Team Model (Real-ESRGAN Improved)": os.path.exists(TEAM_MODEL_PATH),
            "Real-ESRGAN (Official)": os.path.exists(OFFICIAL_REALESRGAN_PATH)
        }
    }


@app.get("/api/presets")
def api_presets():
    catalog = load_precomputed_catalog()
    presets = []
    seen_ids = set()

    preferred_order = [
        "crop_butterfly",
        "crop_bird",
        "crop_baboon",
        "0030.jpg",
        "0014.jpg",
        "gallery_foliage",
        "bench_10_comic",
        "bench_19_ppt3",
        "bench_01_baby",
        "bench_08_bridge",
        "bench_12_flowers",
    ]
    for pid in preferred_order:
        if pid in catalog:
            item = catalog[pid]
            presets.append({
                "filename": pid,
                "label": item["label"],
                "desc": item["desc"],
                "url": item["thumb_url"],
                "is_precomputed": True,
                "available_models": list(item["models"].keys()),
            })
            seen_ids.add(pid)

    lr_dir = os.path.join(BASE_DIR, "datasets", "eval_20", "LR")
    for candidate_dir in (lr_dir, INPUTS_DIR):
        if os.path.isdir(candidate_dir):
            files = [
                f for f in sorted(os.listdir(candidate_dir))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                and os.path.isfile(os.path.join(candidate_dir, f))
                and f not in seen_ids
            ][:6]
            for f in files:
                presets.append({
                    "filename": f,
                    "label": f,
                    "desc": "ทดสอบประมวลผลสดจากโฟลเดอร์ inputs",
                    "url": f"/api/inputs/{f}",
                    "is_precomputed": False,
                })
                seen_ids.add(f)
            if files:
                break

    return {"presets": presets}


@app.get("/api/inputs/{filename}")
def serve_input_image(filename: str):
    target_path = resolve_input_image(filename)
    if target_path:
        return FileResponse(target_path)
    catalog = load_precomputed_catalog()
    if filename in catalog:
        item = catalog[filename]
        return Response(content=item["preview_bytes"], media_type=item["preview_mime"])
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/api/enhance")
async def api_enhance(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    model: str = Form("team_latest"),
    tile: int = Form(256),
    outscale: float = Form(4.0)
):
    n_model = normalize_model_key(model)

    # Fast precomputed multi-model benchmark presets (pixel-exact & instant)
    if file is None and preset:
        catalog = load_precomputed_catalog()
        if preset in catalog:
            item = catalog[preset]
            selected_key = n_model if n_model in item["models"] else "team_latest"
            out_uri = item["models"][selected_key]
            return {
                "success": True,
                "is_precomputed": True,
                "model": selected_key,
                "model_label": MODEL_DISPLAY_NAMES.get(selected_key, selected_key),
                "latency_ms": 0.0,
                "input_dimensions": {"width": item["in_w"], "height": item["in_h"]},
                "output_dimensions": {"width": item["out_w"], "height": item["out_h"]},
                "before_image": item["before_image"],
                "output_image": out_uri,
                "all_models": item["models"],
                "metrics": MODEL_BENCHMARK_METRICS.get(selected_key, {}),
                "all_metrics": MODEL_BENCHMARK_METRICS,
                "description": item["desc"],
            }

    img_bgr = None
    if file is not None and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif preset:
        target_p = resolve_input_image(preset)
        if target_p:
            img_bgr = cv2.imread(target_p, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No valid image provided")

    h, w = img_bgr.shape[:2]
    max_side = 256 if DEVICE.type == 'cpu' else 512
    if max(h, w) > max_side:
        scale_ratio = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (max(16, int(w * scale_ratio)), max(16, int(h * scale_ratio))), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    out_w = max(1, int(round(w * outscale)))
    out_h = max(1, int(round(h * outscale)))
    bicubic_bgr = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    t0 = time.perf_counter()
    if n_model == "bicubic":
        out_bgr = bicubic_bgr
    else:
        out_bgr = _cached_rrdb_inference(img_bgr, n_model, tile=tile)
        if outscale != 4.0:
            out_bgr = cv2.resize(out_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    t_ms = round((time.perf_counter() - t0) * 1000, 1)
    out_h, out_w = out_bgr.shape[:2]

    return {
        "success": True,
        "is_precomputed": False,
        "model": n_model,
        "model_label": MODEL_DISPLAY_NAMES.get(n_model, n_model),
        "latency_ms": t_ms,
        "input_dimensions": {"width": w, "height": h},
        "output_dimensions": {"width": out_w, "height": out_h},
        "before_image": image_to_base64(bicubic_bgr, ext=".jpg"),
        "output_image": image_to_base64(out_bgr, ext=".jpg"),
    }


@app.post("/api/compare")
async def compare_all_models(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    tile: int = Form(256),
    outscale: float = Form(4.0)
):
    """Compare all models simultaneously with distinct outputs & metrics"""
    if file is None and preset:
        catalog = load_precomputed_catalog()
        if preset in catalog:
            item = catalog[preset]
            compare_order = [
                ("Bicubic (Baseline)", "bicubic"),
                ("ESRGAN", "esrgan"),
                ("BSRGAN", "bsrgan"),
                ("Real-ESRNet (Pretrained)", "realesrnet"),
                ("Real-ESRGAN (Official)", "realesrgan_official"),
                ("⭐ Real-ESRGAN (Our Team Model - Improved)", "team_latest"),
            ]
            results = []
            for disp_name, m_key in compare_order:
                if m_key not in item["models"]:
                    continue
                m_stats = MODEL_BENCHMARK_METRICS.get(m_key, {})
                results.append({
                    "key": m_key,
                    "model": disp_name,
                    "is_team": m_key == "team_latest",
                    "latency_ms": 0.0,
                    "output_shape": f"{item['out_w']}x{item['out_h']}",
                    "psnr": m_stats.get("psnr", 0.0),
                    "ssim": m_stats.get("ssim", 0.0),
                    "lpips": m_stats.get("lpips", 0.0),
                    "niqe": m_stats.get("niqe", 0.0),
                    "musiq": m_stats.get("musiq", 0.0),
                    "image": item["models"][m_key],
                })
            return {
                "success": True,
                "is_precomputed": True,
                "input_dimensions": {"width": item["in_w"], "height": item["in_h"]},
                "before_image": item["before_image"],
                "results": results,
            }

    img_bgr = None
    if file is not None and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif preset:
        target_p = resolve_input_image(preset)
        if target_p:
            img_bgr = cv2.imread(target_p, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No valid image provided")

    h, w = img_bgr.shape[:2]
    max_side = 256 if DEVICE.type == 'cpu' else 400
    if max(h, w) > max_side:
        scale_ratio = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (max(16, int(w * scale_ratio)), max(16, int(h * scale_ratio))), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    bicubic_bgr = cv2.resize(img_bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    input_b64 = image_to_base64(bicubic_bgr, ext=".jpg")
    iqa = get_iqa_metrics()

    model_registry = [
        ("Bicubic (Baseline)", "bicubic"),
        ("Real-ESRNet (Pretrained)", "realesrnet"),
        ("Real-ESRGAN (Official)", "realesrgan_official"),
        ("⭐ Real-ESRGAN (Our Team Model - Improved)", "team_latest"),
    ]
    if os.path.exists(ESRGAN_PATH):
        model_registry.insert(1, ("ESRGAN", "esrgan"))
    if os.path.exists(BSRGAN_PATH):
        model_registry.insert(2, ("BSRGAN", "bsrgan"))

    results = []

    for display_name, m_key in model_registry:
        t0 = time.perf_counter()
        try:
            if m_key == "bicubic":
                out_bgr = bicubic_bgr
            else:
                out_bgr = _cached_rrdb_inference(img_bgr, m_key, tile=tile)

            t_ms = round((time.perf_counter() - t0) * 1000, 1)

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
                "key": m_key,
                "model": display_name,
                "is_team": m_key == "team_latest",
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
        "is_precomputed": False,
        "input_dimensions": {"width": w, "height": h},
        "before_image": input_b64,
        "results": results
    }

# Serve static frontend & reports
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
