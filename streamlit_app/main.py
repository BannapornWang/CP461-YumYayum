from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = ROOT_DIR / "webapp" / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rrdbnet import RRDBNet  # noqa: E402

# Limit CPU threads to prevent CPU quota throttling on Streamlit Cloud
torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

st.set_page_config(
    page_title="YumYayum Super-Resolution Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "Real-ESRGAN (Official)": {
        "key": "realesrgan_official",
        "local": (ROOT_DIR / "weights" / "RealESRGAN_x4plus.pth",),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "Real-ESRNet": {
        "key": "realesrnet",
        "local": (ROOT_DIR / "weights" / "RealESRNet_x4plus.pth",),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    },
    "ESRGAN": {
        "key": "esrgan",
        "local": (ROOT_DIR / "weights" / "ESRGAN_x4.pth",),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    },
    "BSRGAN": {
        "key": "bsrgan",
        "local": (ROOT_DIR / "weights" / "BSRGAN.pth",),
        "url": None,
        "bsrgan": True,
    },
    "⭐ Real-ESRGAN (Our Team Model - Improved)": {
        "key": "team",
        "local": (
            ROOT_DIR / "weights" / "RealESRGAN_OurTeam_Improved.pth",
            ROOT_DIR / "RealESRGAN_OurTeam_Improved.pth",
            ROOT_DIR / "net_g_latest.pth",
            ROOT_DIR / "net_g_52000.pth",
            ROOT_DIR / "experiments" / "train_RealESRNetx4_CP461_team_5h" / "models" / "net_g_9000.pth",
        ),
        "url": None,
    },
}

REPORTS = {
    "📊 ผลการทดลองและการวิเคราะห์ (Academic Benchmark)": "benchmark_results.html",
    "📈 ภาพรวมและสถิติชุดข้อมูล (EDA Dashboard)": "eda_dashboard.html",
    "📋 ตารางเปรียบเทียบมาตรฐาน 20 ภาพ": "model_comparison_20_table.html",
    "🖼️ แกลเลอรีผลลัพธ์โมเดล (Model Gallery)": "model_comparison_gallery.html",
}


def usable_checkpoint(path: Path) -> bool:
    """Ignore absent files and small Git LFS pointer files."""
    try:
        return path.is_file() and path.stat().st_size > 1024 * 1024
    except OSError:
        return False


def local_checkpoint(spec: dict[str, Any]) -> Path | None:
    for candidate in spec["local"]:
        if usable_checkpoint(candidate):
            return candidate
    return None


def available_model_names(uploaded_checkpoint: Path | None) -> list[str]:
    names = ["Bicubic (Baseline)"]
    for name, spec in MODEL_SPECS.items():
        if local_checkpoint(spec) is not None or spec.get("url"):
            names.append(name)
    if uploaded_checkpoint is not None:
        names.append("Uploaded team checkpoint")
    return names


def save_uploaded_checkpoint(upload: Any) -> Path:
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    target_dir = Path(tempfile.gettempdir()) / "yumyayum-checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}.pth"
    if not target.exists():
        target.write_bytes(data)
    return target


def download_checkpoint(url: str) -> Path:
    cache_dir = Path.home() / ".cache" / "yumyayum-models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / url.rsplit("/", 1)[-1]
    if usable_checkpoint(target):
        return target

    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "YumYayum-Streamlit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not usable_checkpoint(partial):
            raise RuntimeError("The downloaded checkpoint is empty or incomplete.")
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return target


def convert_bsrgan_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    replacements = (
        ("RRDB_trunk.", "body."),
        (".RDB1.", ".rdb1."),
        (".RDB2.", ".rdb2."),
        (".RDB3.", ".rdb3."),
        ("trunk_conv.", "conv_body."),
        ("HRconv.", "conv_hr."),
    )
    converted = {}
    for key, value in state_dict.items():
        for old, new in replacements:
            key = key.replace(old, new)
        converted[key] = value
    return converted


def load_model(checkpoint: Path, is_bsrgan: bool = False) -> torch.nn.Module:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    checkpoint_data = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data.get("params_ema", checkpoint_data.get("params", checkpoint_data))
    else:
        state_dict = checkpoint_data

    if is_bsrgan:
        model.load_state_dict(convert_bsrgan_keys(state_dict), strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model.eval().to(DEVICE)


def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    image_rgb = image_tensor.squeeze(0).detach().float().cpu().clamp_(0, 1).numpy()
    image_rgb = (image_rgb.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def image_png_bytes(image_bgr: np.ndarray) -> bytes:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output = io.BytesIO()
    Image.fromarray(image_rgb).save(output, format="PNG")
    return output.getvalue()


def resize_input(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if max(height, width) <= max_side:
        return image_bgr
    scale = max_side / max(height, width)
    return cv2.resize(
        image_bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def comparison_signature(image_bytes: bytes, selected_models: list[str], max_side: int, include_iqa: bool, checkpoint: Path | None) -> str:
    digest = hashlib.sha256(image_bytes)
    digest.update(repr((selected_models, max_side, include_iqa, checkpoint)).encode())
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def get_iqa_metrics():
    import pyiqa

    return {
        name: pyiqa.create_metric(name, device=DEVICE)
        for name in ("niqe", "musiq")
    }


def run_comparison(
    image_bgr: np.ndarray,
    selected_models: list[str],
    uploaded_checkpoint: Path | None,
    include_iqa: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    height, width = image_bgr.shape[:2]
    input_tensor = to_tensor(image_bgr)
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for name in selected_models:
        started = time.perf_counter()
        try:
            if name == "Bicubic (Baseline)":
                output = cv2.resize(
                    image_bgr,
                    (width * 4, height * 4),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                if name == "Uploaded team checkpoint":
                    checkpoint = uploaded_checkpoint
                    is_bsrgan = False
                else:
                    spec = MODEL_SPECS[name]
                    checkpoint = local_checkpoint(spec)
                    if checkpoint is None and spec.get("url"):
                        checkpoint = download_checkpoint(spec["url"])
                    is_bsrgan = bool(spec.get("bsrgan"))
                if checkpoint is None:
                    raise FileNotFoundError("No checkpoint is available for this model.")

                model = load_model(checkpoint, is_bsrgan=is_bsrgan)
                try:
                    with torch.inference_mode():
                        output = tensor_to_bgr(model(input_tensor))
                finally:
                    del model

            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            result = {
                "model": name,
                "latency_ms": elapsed_ms,
                "width": output.shape[1],
                "height": output.shape[0],
                "image": output,
            }
            if include_iqa:
                try:
                    metrics = get_iqa_metrics()
                    metric_input = to_tensor(output)
                    with torch.inference_mode():
                        niqe_score = round(float(metrics["niqe"](metric_input).item()), 3)
                        musiq_score = round(float(metrics["musiq"](metric_input).item()), 2)
                    result["niqe"] = niqe_score
                    result["musiq"] = musiq_score
                except Exception as error:
                    failures.append(f"NIQE/MUSIQ unavailable: {error}")
                    include_iqa = False
            results.append(result)
        except Exception as error:  # Keep successful model outputs visible.
            failures.append(f"{name}: {error}")

    return results, failures


def list_sample_images() -> list[Path]:
    """Discover preset sample images shipped with the repository."""
    candidates: list[Path] = []
    for folder in (ROOT_DIR / "datasets" / "eval_20" / "LR", ROOT_DIR / "inputs"):
        if folder.is_dir():
            files = [
                path for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if files:
                candidates.extend(files[:8])
                break
    return candidates


def render_report(report_name: str) -> None:
    report_path = STATIC_DIR / REPORTS[report_name]
    if not report_path.is_file():
        st.error(f"ไม่พบไฟล์รายงาน: {report_path.name}")
        return
    import streamlit.components.v1 as components

    height = 2200 if "Academic Benchmark" in report_name else (1400 if "EDA" in report_name or "Gallery" in report_name else 1000)
    components.html(report_path.read_text(encoding="utf-8"), height=height, scrolling=True)


def render_studio() -> None:
    st.title("⚡ YumYayum Super-Resolution Studio")
    st.write("เปรียบเทียบการขยายภาพความละเอียดสูง 4× จากโมเดลของทีมและโมเดลมาตรฐานแบบเคียงข้างกัน")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.caption(f"อุปกรณ์ประมวลผล (Inference device): **{device_name}**")

    sample_files = list_sample_images()

    with st.sidebar:
        st.subheader("1. เลือกภาพทดสอบ")
        input_modes = ["🖼️ เลือกภาพตัวอย่างที่มีให้", "📤 อัปโหลดภาพของคุณเอง"] if sample_files else ["📤 อัปโหลดภาพของคุณเอง"]
        input_mode = st.radio("แหล่งที่มาของภาพ:", input_modes, index=0)

        image_bytes: bytes | None = None
        image_label = "input"
        if input_mode == "🖼️ เลือกภาพตัวอย่างที่มีให้" and sample_files:
            sample_map = {path.name: path for path in sample_files}
            selected_sample = st.selectbox("เลือกภาพตัวอย่างจากโปรเจกต์:", list(sample_map.keys()))
            sample_path = sample_map[selected_sample]
            image_bytes = sample_path.read_bytes()
            image_label = sample_path.name
        else:
            image_upload = st.file_uploader(
                "อัปโหลดภาพ (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                help="ภาพขนาดใหญ่จะถูกปรับขนาดชั่วคราวก่อนขยาย เพื่อป้องกันหน่วยความจำเต็ม",
            )
            if image_upload is not None:
                image_bytes = image_upload.getvalue()
                image_label = image_upload.name

        st.subheader("2. เลือกโมเดลเปรียบเทียบ")
        with st.expander("⚙️ ตั้งค่าขั้นสูง (Advanced Settings)", expanded=False):
            max_side = st.slider(
                "ขนาดด้านที่ยาวสุดของภาพเข้า (px)",
                min_value=96,
                max_value=320,
                value=160,
                step=32,
                help="ค่าแนะนำสำหรับ Cloud CPU คือ 160 px เพื่อให้รันเสร็จเร็ว",
            )
            checkpoint_upload = st.file_uploader(
                "อัปโหลด Checkpoint (.pth) เพิ่มเติม",
                type=["pth", "pt"],
                help="ใช้เมื่อต้องการทดสอบไฟล์โมเดลอื่นนอกเหนือจากที่มีในโปรเจกต์",
            )
            include_iqa = st.checkbox(
                "คำนวณคะแนน NIQE และ MUSIQ",
                value=False,
                help="ตัวเลือกเสริม (ต้องติดตั้งแพ็กเกจใน requirements-iqa.txt)",
            )

        uploaded_checkpoint = save_uploaded_checkpoint(checkpoint_upload) if checkpoint_upload else None
        model_names = available_model_names(uploaded_checkpoint)
        preferred_defaults = [
            name for name in (
                "Bicubic (Baseline)",
                "⭐ Real-ESRGAN (Our Team Model - Improved)",
                "Real-ESRGAN (Official)",
            )
            if name in model_names
        ]
        default_models = preferred_defaults[:2] if len(preferred_defaults) >= 2 else preferred_defaults
        selected_models = st.multiselect(
            "โมเดลที่ต้องการเทียบ:",
            options=model_names,
            default=default_models,
        )

    team_spec = MODEL_SPECS.get("⭐ Real-ESRGAN (Our Team Model - Improved)") or next((v for v in MODEL_SPECS.values() if v.get("key") == "team"), None)
    if team_spec and local_checkpoint(team_spec) is None and uploaded_checkpoint is None:
        st.info("ไม่พบไฟล์น้ำหนักโมเดลของทีมในโฟลเดอร์ สามารถอัปโหลดไฟล์ .pth ได้ในเมนู '⚙️ ตั้งค่าขั้นสูง' ทางซ้าย")
    if image_bytes is None:
        st.info("กรุณาเลือกภาพตัวอย่าง หรืออัปโหลดภาพจากแถบด้านซ้ายเพื่อเริ่มต้น")
        return
    if not selected_models:
        st.warning("กรุณาเลือกโมเดลอย่างน้อย 1 โมเดล")
        return

    try:
        image_rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr = resize_input(image_bgr, max_side)
    except Exception as error:
        st.error(f"ไม่สามารถอ่านไฟล์ภาพนี้ได้: {error}")
        return

    source_col, settings_col = st.columns([2, 1])
    with source_col:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption=f"ภาพต้นฉบับ ({image_label}) — {image_bgr.shape[1]} × {image_bgr.shape[0]} px",
            use_container_width=True,
        )
    with settings_col:
        st.metric("อัตราขยายภาพ (Scale)", "4×")
        st.metric("จำนวนโมเดลที่เลือก", f"{len(selected_models)} โมเดล")
        st.caption(
            "กดปุ่มด้านล่างเพื่อขยายภาพและเทียบผลลัพธ์ทีละโมเดลแบบเคียงข้างกับภาพต้นฉบับ "
            "(โมเดล Official จะดาวน์โหลดอัตโนมัติในการใช้งานครั้งแรก)"
        )

    if st.button("🚀 เริ่มขยายภาพและเปรียบเทียบ (Run Comparison)", type="primary", use_container_width=True):
        with st.spinner("กำลังประมวลผลโมเดลที่เลือก... (หากรันบน CPU อาจใช้เวลาสักครู่ต่อโมเดล)"):
            results, failures = run_comparison(image_bgr, selected_models, uploaded_checkpoint, include_iqa)
        st.session_state["comparison_results"] = results
        st.session_state["comparison_failures"] = failures
        st.session_state["comparison_signature"] = comparison_signature(
            image_bytes, selected_models, max_side, include_iqa, uploaded_checkpoint
        )

    current_signature = comparison_signature(
        image_bytes, selected_models, max_side, include_iqa, uploaded_checkpoint
    )
    is_current = st.session_state.get("comparison_signature") == current_signature
    results = st.session_state.get("comparison_results", []) if is_current else []
    failures = st.session_state.get("comparison_failures", []) if is_current else []
    for failure in failures:
        st.warning(failure)
    if not results:
        return

    st.subheader("📊 ผลการเปรียบเทียบ (Comparison Results)")
    st.dataframe(
        [
            {
                "โมเดล (Model)": item["model"],
                "เวลาประมวลผล (ms)": item["latency_ms"],
                "ขนาดภาพ 4×": f'{item["width"]} × {item["height"]}',
                **({"NIQE ↓": item["niqe"], "MUSIQ ↑": item["musiq"]} if "niqe" in item else {}),
            }
            for item in results
        ],
        hide_index=True,
        use_container_width=True,
    )

    tabs = st.tabs([item["model"] for item in results])
    input_rgb_preview = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    for tab, item in zip(tabs, results):
        with tab:
            col_before, col_after = st.columns(2)
            with col_before:
                st.image(
                    input_rgb_preview,
                    caption=f"ภาพต้นฉบับ (Input) — {image_bgr.shape[1]} × {image_bgr.shape[0]} px",
                    use_container_width=True,
                )
            with col_after:
                st.image(
                    cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB),
                    caption=f'{item["model"]} (4×) — {item["width"]} × {item["height"]} px · {item["latency_ms"]} ms',
                    use_container_width=True,
                )
            download_name = item["model"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            st.download_button(
                "💾 ดาวน์โหลดภาพ PNG",
                data=image_png_bytes(item["image"]),
                file_name=f"{download_name}_x4.png",
                mime="image/png",
                key=f'download-{item["model"]}',
            )


def main() -> None:
    # Custom CSS for clean dark theme & Thai readability
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'IBM Plex Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .stApp {
                background-color: #0b0f19;
                color: #f1f5f9;
            }
            [data-testid="stSidebar"] {
                background-color: #0f172a;
                border-right: 1px solid #1e293b;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #f8fafc !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("⚡ YumYayum Studio")
        st.caption("CP461 Computer Vision · Real-ESRGAN Super-Resolution (4×)")
        page_options = [
            "📊 ผลการทดลองและการวิเคราะห์ (Academic Benchmark)",
            "⚡ ทดสอบขยายภาพสด (Live Model Comparison)",
            "📈 ภาพรวมและสถิติชุดข้อมูล (EDA Dashboard)",
            "📋 ตารางเปรียบเทียบมาตรฐาน 20 ภาพ",
            "🖼️ แกลเลอรีผลลัพธ์โมเดล (Model Gallery)",
        ]
        page = st.radio("เลือกหน้าแสดงผล:", page_options, index=0)
    
    if page == "⚡ ทดสอบขยายภาพสด (Live Model Comparison)":
        render_studio()
    else:
        render_report(page)


if __name__ == "__main__":
    main()
