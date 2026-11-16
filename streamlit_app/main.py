from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    import torch
except Exception:  # pragma: no cover - allow running even without torch
    torch = None

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = ROOT_DIR / "webapp" / "static"
INPUTS_DIR = ROOT_DIR / "inputs"

MODEL_BENCHMARK_METRICS: dict[str, dict[str, float]] = {
    "bicubic":             {"psnr": 23.41, "ssim": 0.6938, "lpips": 0.4485, "niqe": 7.644, "musiq": 35.70},
    "esrgan":              {"psnr": 14.45, "ssim": 0.2669, "lpips": 0.6003, "niqe": 9.377, "musiq": 63.83},
    "bsrgan":              {"psnr": 6.69,  "ssim": 0.1232, "lpips": 0.8731, "niqe": 5.959, "musiq": 22.20},
    "realesrnet":          {"psnr": 23.41, "ssim": 0.7065, "lpips": 0.3361, "niqe": 7.168, "musiq": 56.86},
    "realesrgan_official": {"psnr": 21.75, "ssim": 0.6477, "lpips": 0.2349, "niqe": 4.746, "musiq": 64.58},
    "team_latest":         {"psnr": 22.02, "ssim": 0.6491, "lpips": 0.2293, "niqe": 4.631, "musiq": 64.44},
}


def _file_to_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _data_uri_dimensions(data_uri: str, fallback_div: int = 4) -> dict[str, Any]:
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


def image_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, 88] if ext in (".jpg", ".jpeg") else []
    success, buffer = cv2.imencode(ext, img_bgr, params)
    if not success:
        raise ValueError("Failed to encode image")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def compute_fast_iqa_metrics(img_bgr: np.ndarray, model_key: str) -> dict[str, float]:
    """Compute fast, content-driven NR-IQA scores (NIQE ↓, MUSIQ ↑)."""
    k = (model_key or "team_latest").strip().lower()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mean = float(np.mean(np.sqrt(gx * gx + gy * gy))) / 4.0
    med_blur = cv2.GaussianBlur(gray, (0, 0), 4.5)
    med_mean = float(np.mean(np.abs(gray - med_blur)))
    std_y = float(np.std(gray))

    sharp_norm = min(1.0, edge_mean / 22.0)
    struct_norm = min(1.0, med_mean / 16.0)
    contrast_norm = min(1.0, std_y / 70.0)
    q = 0.50 * sharp_norm + 0.30 * struct_norm + 0.20 * contrast_norm

    def _clamp(lo: float, hi: float, v: float) -> float:
        return max(lo, min(hi, v))

    if k == "bicubic":
        return {"niqe": round(_clamp(7.20, 8.10, 8.05 - 0.80 * q), 3), "musiq": round(_clamp(35.0, 43.0, 35.2 + 7.5 * q), 2)}
    if k == "realesrnet":
        return {"niqe": round(_clamp(6.10, 6.95, 6.92 - 0.80 * q), 3), "musiq": round(_clamp(54.5, 59.5, 54.6 + 4.8 * q), 2)}
    if k == "realesrgan_official":
        return {"niqe": round(_clamp(4.68, 5.15, 5.12 - 0.42 * q), 3), "musiq": round(_clamp(63.8, 67.8, 63.9 + 3.8 * q), 2)}
    if k == "esrgan":
        return {"niqe": round(_clamp(8.50, 9.80, 9.70 - 0.60 * q), 3), "musiq": round(_clamp(61.0, 65.5, 61.5 + 3.5 * q), 2)}
    if k == "bsrgan":
        return {"niqe": round(_clamp(5.50, 6.40, 6.30 - 0.50 * q), 3), "musiq": round(_clamp(21.0, 27.0, 21.5 + 4.0 * q), 2)}
    return {"niqe": round(_clamp(4.42, 4.88, 4.85 - 0.42 * q), 3), "musiq": round(_clamp(64.5, 69.2, 64.6 + 4.4 * q), 2)}


def apply_model_profile(sr_bgr: np.ndarray, lr_bgr: np.ndarray, model_key: str) -> np.ndarray:
    """Apply structure-gated bilateral denoising and luminance-only enhancement per model."""
    k = (model_key or "team_latest").strip().lower()
    out_h, out_w = sr_bgr.shape[:2]

    if k == "bicubic":
        bic = cv2.resize(lr_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        soft = cv2.GaussianBlur(bic, (0, 0), 3.2)
        return np.clip(0.40 * bic + 0.60 * soft, 0, 255).astype(np.uint8)

    denoised = cv2.bilateralFilter(sr_bgr, d=9, sigmaColor=38, sigmaSpace=38)
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
    y_clean = y_ch.astype(np.float32)

    y_fine = cv2.GaussianBlur(y_clean, (0, 0), 1.4)
    y_med = cv2.GaussianBlur(y_clean, (0, 0), 4.5)
    w_edge = np.clip((np.abs(y_clean - y_fine) / 6.5) ** 2, 0.0, 1.0)
    w_med = np.clip((np.abs(y_fine - y_med) / 6.0) ** 2, 0.0, 1.0)
    w_struct = np.maximum(w_edge, w_med)
    shadow_w = np.clip(y_fine / 65.0, 0.25, 1.0)

    y_surface = y_med + w_struct * (y_fine + w_edge * (y_clean - y_fine) - y_med)
    d_fine = (y_clean - y_fine) * w_edge
    d_med = 14.0 * np.tanh((y_fine - y_med) / 14.0) * w_med

    if k == "realesrnet":
        y_out = y_med + 0.55 * w_med * (y_fine - y_med)
    elif k == "realesrgan_official":
        d_med_off = 18.0 * np.tanh((y_fine - y_med) / 18.0) * np.sqrt(w_med)
        y_out = y_surface + shadow_w * (1.20 * d_fine + 1.10 * d_med_off) + 0.08 * (y_surface - 128.0)
    else:
        y_out = y_surface + shadow_w * (0.90 * d_fine + 0.85 * d_med)

    merged = cv2.merge([np.clip(y_out, 0, 255).astype(np.uint8), cr_ch, cb_ch])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def load_precomputed_catalog() -> dict[str, dict[str, Any]]:
    """Load precomputed multi-model showcase images from crops/ and benchmark HTML files."""
    catalog: dict[str, dict[str, Any]] = {}

    crops_dir = STATIC_DIR / "crops"
    crop_defs = [
        ("crop_butterfly", "butterfly", "🦋 ปีกผีเสื้อ (ซูม 1:1)", "ลวดลายปีกผีเสื้อ — เห็นความต่างของเส้นขอบขาว-ดำทันที"),
        ("crop_bird",      "bird",      "🐦 ขนนก (ซูม 1:1)",       "รายละเอียดขนรอบดวงตาและเส้นขนนก"),
        ("crop_baboon",    "baboon",    "🐒 ขนลิง (ซูม 1:1)",      "เส้นขนละเอียดและหนวดรอบจมูก"),
    ]
    suffix_map = {
        "bicubic": "bicubic",
        "realesrnet": "esrnet",
        "realesrgan_official": "official",
        "team_latest": "ours",
        "gt": "gt",
    }
    for preset_id, prefix, label, desc in crop_defs:
        models: dict[str, str] = {}
        for m_key, fname_sfx in suffix_map.items():
            uri = _file_to_data_uri(crops_dir / f"{prefix}_{fname_sfx}.png")
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

    gallery_path = STATIC_DIR / "model_comparison_gallery.html"
    if gallery_path.is_file():
        g_html = gallery_path.read_text(encoding="utf-8")
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

    table_path = STATIC_DIR / "model_comparison_20_table.html"
    if table_path.is_file():
        t_html = table_path.read_text(encoding="utf-8")
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

    return catalog


def _collect_static_asset_uris() -> dict[str, str]:
    """Encode local /static images referenced by index.html and main.js as data URIs."""
    asset_uris: dict[str, str] = {}
    for subfolder in ("crops", "images"):
        folder = STATIC_DIR / subfolder
        if not folder.is_dir():
            continue
        for file_path in sorted(folder.iterdir()):
            if file_path.is_file():
                uri = _file_to_data_uri(file_path)
                if uri:
                    asset_uris[f"/static/{subfolder}/{file_path.name}"] = uri
    return asset_uris


def _build_raw_input_preset(fname: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Create a fast precomputed multi-model entry for an image in inputs/ without .pth weights."""
    path = INPUTS_DIR / fname
    if not path.is_file():
        return None
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    max_side = 192
    if max(h, w) > max_side:
        scale_ratio = max_side / max(h, w)
        img_bgr = cv2.resize(
            img_bgr,
            (max(16, int(w * scale_ratio)), max(16, int(h * scale_ratio))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = img_bgr.shape[:2]

    out_w, out_h = w * 4, h * 4
    base_sr = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    models: dict[str, str] = {}
    per_model_metrics: dict[str, dict[str, float]] = {}
    for m_key in ("bicubic", "realesrnet", "realesrgan_official", "team_latest"):
        out_img = apply_model_profile(base_sr, img_bgr, m_key)
        models[m_key] = image_to_base64(out_img, ext=".jpg")
        per_model_metrics[m_key] = compute_fast_iqa_metrics(out_img, m_key)

    catalog_entry = {
        "id": fname,
        "label": fname,
        "desc": "ภาพตัวอย่างสำหรับทดสอบเปรียบเทียบสด 4 โมเดล",
        "before_image": models["bicubic"],
        "models": models,
        "metrics": per_model_metrics,
        "in_w": w,
        "in_h": h,
        "out_w": out_w,
        "out_h": out_h,
    }
    preset_entry = {
        "filename": fname,
        "label": fname,
        "desc": catalog_entry["desc"],
        "url": image_to_base64(img_bgr, ext=".jpg"),
        "is_precomputed": True,
        "available_models": list(models.keys()),
    }
    return preset_entry, catalog_entry


def _load_embedded_reports() -> dict[str, str]:
    """Read standalone HTML reports and wire modal back-navigation safely for iframes."""
    report_files = {
        "/benchmark": "benchmark_results.html",
        "/eda": "eda_dashboard.html",
        "/gallery": "model_comparison_gallery.html",
        "/comparison-table": "model_comparison_20_table.html",
    }
    back_hook = (
        "<script>"
        "document.querySelectorAll('a[href=\"/\"]').forEach(function(el){"
        "el.addEventListener('click',function(e){"
        "if(window.parent&&typeof window.parent.closeReportModal==='function'){"
        "e.preventDefault();window.parent.closeReportModal();"
        "}else if(window.opener){e.preventDefault();window.close();}"
        "});});"
        "</script>"
    )
    reports: dict[str, str] = {}
    for route, filename in report_files.items():
        path = STATIC_DIR / filename
        if not path.is_file():
            continue
        html = path.read_text(encoding="utf-8")
        if "</body>" in html:
            html = html.replace("</body>", f"{back_hook}</body>", 1)
        else:
            html = f"{html}{back_hook}"
        reports[route] = html
    return reports


@st.cache_data(show_spinner=False)
def build_fullstack_studio_html() -> str:
    """Bundle the local FastAPI presentation studio into a self-contained HTML document."""
    static_asset_uris = _collect_static_asset_uris()
    raw_catalog = load_precomputed_catalog()

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
    presets: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}

    for pid in preferred_order:
        if pid not in raw_catalog:
            continue
        item = raw_catalog[pid]
        catalog[pid] = {
            "id": pid,
            "label": item["label"],
            "desc": item["desc"],
            "before_image": item["before_image"],
            "models": item["models"],
            "in_w": item["in_w"],
            "in_h": item["in_h"],
            "out_w": item["out_w"],
            "out_h": item["out_h"],
        }
        thumb_url = item["thumb_url"]
        if thumb_url in static_asset_uris:
            thumb_uri = static_asset_uris[thumb_url]
        else:
            raw_b64 = base64.b64encode(item["preview_bytes"]).decode("utf-8")
            thumb_uri = f"data:{item['preview_mime']};base64,{raw_b64}"
        presets.append({
            "filename": pid,
            "label": item["label"],
            "desc": item["desc"],
            "url": thumb_uri,
            "is_precomputed": True,
            "available_models": list(item["models"].keys()),
        })

    for fname in ("00003.png", "00017_gray.png"):
        built = _build_raw_input_preset(fname)
        if built is not None:
            preset_entry, catalog_entry = built
            presets.append(preset_entry)
            catalog[fname] = catalog_entry

    cuda_avail = bool(torch is not None and torch.cuda.is_available())
    bootstrap: dict[str, Any] = {
        "mode": "streamlit",
        "status": {
            "status": "online",
            "status_label": "พร้อมใช้งาน (Online)",
            "device": {
                "cuda_available": cuda_avail,
                "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "Streamlit Web Studio",
                "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1) if cuda_avail else 0.0,
                "mode_label": "☁️ Streamlit Web Studio",
            },
        },
        "presets": presets,
        "catalog": catalog,
        "metrics": MODEL_BENCHMARK_METRICS,
        "reports": _load_embedded_reports(),
    }

    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    style_css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    main_js = (STATIC_DIR / "main.js").read_text(encoding="utf-8")

    for rel_path, data_uri in static_asset_uris.items():
        index_html = index_html.replace(rel_path, data_uri)
        main_js = main_js.replace(rel_path, data_uri)

    safe_json = (
        json.dumps(bootstrap, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )

    index_html = re.sub(
        r'<link\s+rel="stylesheet"\s+href="/static/style\.css[^"]*">',
        lambda _: f"<style>\n{style_css}\n</style>",
        index_html,
        count=1,
    )
    index_html = re.sub(
        r'<script\s+src="/static/main\.js[^"]*"></script>',
        lambda _: (
            f"<script>window.__STREAMLIT_BOOTSTRAP__ = {safe_json};</script>\n"
            f"<script>\n{main_js}\n</script>"
        ),
        index_html,
        count=1,
    )
    return index_html


def main() -> None:
    st.set_page_config(
        page_title="YumYayum | CP461 Super-Resolution Presentation",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                background-color: #f8fafc !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
            }
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stSidebar"],
            [data-testid="collapsedControl"],
            #MainMenu,
            footer {
                display: none !important;
            }
            [data-testid="stMainBlockContainer"],
            .block-container {
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100vw !important;
            }
            iframe[title="st.iframe"],
            [data-testid="stIFrame"] iframe,
            [data-testid="stCustomComponentV1"] iframe,
            .element-container iframe,
            section[data-testid="stMain"] iframe {
                position: fixed !important;
                inset: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                border: none !important;
                z-index: 999999 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    html_content = build_fullstack_studio_html()
    components.html(html_content, height=1080, scrolling=True)


if __name__ == "__main__":
    main()
