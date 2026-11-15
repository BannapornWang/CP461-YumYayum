import os
import sys
import time
import base64
import logging
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import numpy as np
import cv2
import torch

from webapp.app import (
    BASE_DIR,
    DEVICE,
    INPUTS_DIR,
    WEIGHTS_DIR,
    STATIC_DIR,
    TEAM_MODEL_PATH,
    OFFICIAL_REALESRGAN_PATH,
    ESRGAN_PATH,
    BSRGAN_PATH,
    REALESRNET_PATH,
    MODEL_DISPLAY_NAMES,
    MODEL_BENCHMARK_METRICS,
    normalize_model_key,
    load_precomputed_catalog,
    get_iqa_metrics,
    load_rrdb_model,
    apply_model_profile,
    _cached_rrdb_inference,
    resolve_input_image,
    image_to_base64,
    api_presets as _shared_presets_payload,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YumYayumFlask")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)


@app.route("/api/status", methods=["GET"])
def api_status():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if cuda_avail else 0.0
    return jsonify({
        "status": "online",
        "device": {"cuda_available": cuda_avail, "gpu_name": gpu_name, "vram_total_gb": vram_total_gb},
        "available_models": {
            "Bicubic": True,
            "ESRGAN": os.path.exists(ESRGAN_PATH),
            "BSRGAN": os.path.exists(BSRGAN_PATH),
            "Real-ESRNet": os.path.exists(REALESRNET_PATH),
            "Our Team Model (Real-ESRGAN Improved)": os.path.exists(TEAM_MODEL_PATH),
            "Real-ESRGAN (Official)": os.path.exists(OFFICIAL_REALESRGAN_PATH),
        },
    })


def _read_flask_input():
    file = request.files.get("file")
    preset = request.form.get("preset")
    img_bgr = None
    if file and file.filename:
        content = file.read()
        nparr = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif preset:
        target_p = resolve_input_image(preset)
        if target_p:
            img_bgr = cv2.imread(target_p, cv2.IMREAD_COLOR)
    return img_bgr, file, preset


@app.route("/api/compare", methods=["POST"])
def compare_all_models():
    file = request.files.get("file")
    preset = request.form.get("preset")
    tile = int(request.form.get("tile", 256))

    if (not file or not file.filename) and preset:
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
            return jsonify({
                "success": True,
                "is_precomputed": True,
                "input_dimensions": {"width": item["in_w"], "height": item["in_h"]},
                "before_image": item["before_image"],
                "results": results,
            })

    img_bgr, _, _ = _read_flask_input()
    if img_bgr is None:
        return jsonify({"detail": "No valid image provided"}), 400

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
    results = []

    for display_name, m_key in model_registry:
        t0 = time.perf_counter()
        try:
            out_bgr = bicubic_bgr if m_key == "bicubic" else _cached_rrdb_inference(img_bgr, m_key, tile=tile)
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
                "image": image_to_base64(out_bgr, ext=".jpg"),
            })
        except Exception as e:
            logger.error(f"Error evaluating {display_name}: {e}")

    return jsonify({
        "success": True,
        "is_precomputed": False,
        "input_dimensions": {"width": w, "height": h},
        "before_image": input_b64,
        "results": results,
    })


@app.route("/api/presets", methods=["GET"])
def api_presets():
    return jsonify(_shared_presets_payload())


@app.route("/api/inputs/<filename>")
def serve_input_image(filename):
    target_path = resolve_input_image(filename)
    if target_path:
        return send_file(target_path)
    catalog = load_precomputed_catalog()
    if filename in catalog:
        item = catalog[filename]
        return Response(item["preview_bytes"], mimetype=item["preview_mime"])
    return jsonify({"error": "File not found"}), 404


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    file = request.files.get("file")
    preset = request.form.get("preset")
    n_model = normalize_model_key(request.form.get("model", "team_latest"))
    tile = int(request.form.get("tile", 256))
    outscale = float(request.form.get("outscale", 4.0))

    if (not file or not file.filename) and preset:
        catalog = load_precomputed_catalog()
        if preset in catalog:
            item = catalog[preset]
            selected_key = n_model if n_model in item["models"] else "team_latest"
            return jsonify({
                "success": True,
                "is_precomputed": True,
                "model": selected_key,
                "model_label": MODEL_DISPLAY_NAMES.get(selected_key, selected_key),
                "latency_ms": 0.0,
                "input_dimensions": {"width": item["in_w"], "height": item["in_h"]},
                "output_dimensions": {"width": item["out_w"], "height": item["out_h"]},
                "before_image": item["before_image"],
                "output_image": item["models"][selected_key],
                "all_models": item["models"],
                "metrics": MODEL_BENCHMARK_METRICS.get(selected_key, {}),
                "all_metrics": MODEL_BENCHMARK_METRICS,
                "description": item["desc"],
            })

    img_bgr, _, _ = _read_flask_input()
    if img_bgr is None:
        return jsonify({"detail": "No valid image provided"}), 400

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

    return jsonify({
        "success": True,
        "is_precomputed": False,
        "model": n_model,
        "model_label": MODEL_DISPLAY_NAMES.get(n_model, n_model),
        "latency_ms": t_ms,
        "input_dimensions": {"width": w, "height": h},
        "output_dimensions": {"width": out_w, "height": out_h},
        "before_image": image_to_base64(bicubic_bgr, ext=".jpg"),
        "output_image": image_to_base64(out_bgr, ext=".jpg"),
    })

@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/benchmark")
def serve_benchmark():
    return send_from_directory(STATIC_DIR, "benchmark_results.html")

@app.route("/eda")
def serve_eda():
    return send_from_directory(STATIC_DIR, "eda_dashboard.html")

@app.route("/comparison-table")
def serve_comparison_table():
    return send_from_directory(STATIC_DIR, "model_comparison_20_table.html")

@app.route("/gallery")
def serve_gallery():
    return send_from_directory(STATIC_DIR, "model_comparison_gallery.html")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀 YumYayum Super-Resolution Web Studio (Flask Mode)")
    print("  🌐 http://localhost:8000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=8000, debug=False)

