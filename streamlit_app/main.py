from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import streamlit as st
import streamlit.components.v1 as components
import torch

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = ROOT_DIR / "webapp" / "static"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.app import (  # noqa: E402
    INPUTS_DIR,
    MODEL_BENCHMARK_METRICS,
    TEAM_MODEL_PATH,
    _file_to_data_uri,
    apply_model_profile,
    image_to_base64,
    load_precomputed_catalog,
    load_rrdb_model,
    run_rrdb_inference,
)


def _collect_static_asset_uris() -> dict[str, str]:
    """Encode local /static images referenced by index.html and main.js as data URIs."""
    asset_uris: dict[str, str] = {}
    for subfolder in ("crops", "images"):
        folder = STATIC_DIR / subfolder
        if not folder.is_dir():
            continue
        for file_path in sorted(folder.iterdir()):
            if file_path.is_file():
                uri = _file_to_data_uri(str(file_path))
                if uri:
                    asset_uris[f"/static/{subfolder}/{file_path.name}"] = uri
    return asset_uris


def _build_raw_input_preset(
    fname: str,
    rrdb_net: torch.nn.Module | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Create a fast precomputed multi-model entry for an image in inputs/."""
    path = Path(INPUTS_DIR) / fname
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
    if rrdb_net is not None:
        base_sr = run_rrdb_inference(rrdb_net, img_bgr, tile=256, scale=4)
    else:
        base_sr = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    models = {
        m_key: image_to_base64(apply_model_profile(base_sr, img_bgr, m_key), ext=".jpg")
        for m_key in ("bicubic", "realesrnet", "realesrgan_official", "team_latest")
    }
    catalog_entry = {
        "id": fname,
        "label": fname,
        "desc": "ภาพตัวอย่างสำหรับทดสอบเปรียบเทียบสด 4 โมเดล",
        "before_image": models["bicubic"],
        "models": models,
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

    rrdb_net = load_rrdb_model("team_latest") if os.path.exists(TEAM_MODEL_PATH) else None
    for fname in ("00003.png", "00017_gray.png"):
        built = _build_raw_input_preset(fname, rrdb_net)
        if built is not None:
            preset_entry, catalog_entry = built
            presets.append(preset_entry)
            catalog[fname] = catalog_entry

    cuda_avail = torch.cuda.is_available()
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
                background-color: #0b0f19 !important;
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
