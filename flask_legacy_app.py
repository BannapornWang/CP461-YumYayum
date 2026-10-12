import os
import io
import time
import base64
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from basicsr.archs.rrdbnet_arch import RRDBNet
import pyiqa

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Server starting on device: {device}")

# 1. Load Metric Evaluators (No-Reference metrics since user uploads arbitrary images)
print("Loading metrics (NIQE, MUSIQ)...")
nr_metrics = {
    'niqe': pyiqa.create_metric('niqe', device=device),
    'musiq': pyiqa.create_metric('musiq', device=device),
    'psnr': pyiqa.create_metric('psnr', device=device),
    'ssim': pyiqa.create_metric('ssim', device=device),
    'lpips': pyiqa.create_metric('lpips', device=device)
}

# 2. Key mapping for BSRGAN
def convert_bsrgan_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('RRDB_trunk.', 'body.').replace('.RDB1.', '.rdb1.').replace('.RDB2.', '.rdb2.').replace('.RDB3.', '.rdb3.').replace('trunk_conv.', 'conv_body.').replace('HRconv.', 'conv_hr.')
        new_state_dict[new_k] = v
    return new_state_dict

def load_rrdb_model(weights_path, is_bsrgan=False):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
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
    return model.to(device)

print("Preloading Deep Learning Super-Resolution models into GPU...")
models = {}
if os.path.exists('weights/ESRGAN_x4.pth'):
    models['ESRGAN'] = load_rrdb_model('weights/ESRGAN_x4.pth')
if os.path.exists('weights/BSRGAN.pth'):
    models['BSRGAN'] = load_rrdb_model('weights/BSRGAN.pth', is_bsrgan=True)
if os.path.exists('weights/RealESRNet_x4plus.pth'):
    models['Real-ESRNet'] = load_rrdb_model('weights/RealESRNet_x4plus.pth')
if os.path.exists('net_g_52000.pth'):
    models['Real-ESRGAN (Ours 52k)'] = load_rrdb_model('net_g_52000.pth')
if os.path.exists('weights/RealESRGAN_x4plus.pth'):
    models['Real-ESRGAN (Official)'] = load_rrdb_model('weights/RealESRGAN_x4plus.pth')

print("All models loaded successfully!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Super-Resolution Live Comparison Playground</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; font-family: 'Sarabun', sans-serif; }
        body { background: #0b0f19; color: #f8fafc; margin: 0; padding: 24px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 1450px; }
        .header { background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 24px 30px; text-align: center; margin-bottom: 24px; }
        h1 { margin: 0 0 8px 0; font-size: 26px; color: #38bdf8; }
        p.sub { margin: 0; font-size: 15px; color: #94a3b8; }

        /* Upload Area */
        .upload-card {
            background: #111827; border: 2px dashed #374151; border-radius: 16px; padding: 32px 20px;
            text-align: center; cursor: pointer; transition: all 0.2s ease; margin-bottom: 24px;
        }
        .upload-card:hover, .upload-card.dragover { border-color: #38bdf8; background: #172033; }
        .upload-icon { font-size: 40px; margin-bottom: 8px; }
        .upload-text { font-size: 16px; font-weight: 600; color: #e2e8f0; }
        .upload-sub { font-size: 13px; color: #64748b; margin-top: 4px; }
        input[type="file"] { display: none; }

        /* Sample Presets */
        .presets { display: flex; gap: 10px; justify-content: center; align-items: center; margin-top: 14px; flex-wrap: wrap; }
        .preset-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 6px 14px; border-radius: 20px; font-size: 13px; cursor: pointer; transition: all 0.2s; }
        .preset-btn:hover { background: #2563eb; color: #fff; border-color: #2563eb; }

        /* Loading Spinner */
        #loader { display: none; text-align: center; padding: 30px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1); width: 44px; height: 44px; border-radius: 50%;
            border-left-color: #38bdf8; animation: spin 1s linear infinite; margin: 0 auto 14px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Results Display */
        #resultSection { display: none; }
        .gallery-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 30px;
        }
        .model-card {
            background: #111827; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden;
            display: flex; flex-direction: column; transition: all 0.2s ease;
        }
        .model-card:hover { transform: translateY(-4px); border-color: #38bdf8; }
        .model-card.highlight-team { border: 2px solid #3b82f6; background: #172554; }
        .card-header { padding: 12px 14px; text-align: center; font-size: 14px; font-weight: 600; border-bottom: 1px solid #1f2937; }
        .badge-team { background: #2563eb; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 6px; }
        .card-img-wrap { width: 100%; height: 250px; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        .card-img-wrap img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s; cursor: pointer; }
        .card-img-wrap img:hover { transform: scale(1.35); }
        .card-footer { padding: 10px; font-size: 12.5px; color: #94a3b8; display: flex; justify-content: space-around; background: rgba(0,0,0,0.2); }

        /* Table Card */
        .table-card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 24px; margin-bottom: 24px; overflow-x: auto; }
        .table-title { font-size: 18px; font-weight: 600; color: #f8fafc; margin-bottom: 14px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 14.5px; }
        th, td { padding: 12px 16px; }
        th { background: #1f2937; color: #94a3b8; font-weight: 600; border-bottom: 2px solid #374151; }
        td { border-bottom: 1px solid #1f2937; }
        tr:hover td { background: #172033; }
        .tr-team { background: rgba(37,99,235,0.15) !important; font-weight: 600; color: #60a5fa; }
        .val-best { color: #10b981; font-weight: 700; }
        .arrow-up { color: #38bdf8; } .arrow-down { color: #f43f5e; }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Super-Resolution Live Model Comparison Playground</h1>
        <p class="sub">ทดสอบอัปโหลดภาพจริง แล้วระบบจะรันขยายภาพแบบ 4× พร้อมเปรียบเทียบทุกโมเดลและคำนวณ Metrics สดทันที</p>
    </div>

    <!-- Drag & Drop Upload -->
    <div class="upload-card" id="dropArea" onclick="document.getElementById('fileInput').click()">
        <div class="upload-icon">📷</div>
        <div class="upload-text">คลิกเพื่อเลือกภาพ หรือลากไฟล์มาวางที่นี่</div>
        <div class="upload-sub">รองรับ JPG, PNG, WEBP (ภาพจะถูกส่งไปขยาย 4× ด้วย GPU RTX 3060)</div>
        <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
    </div>

    <!-- Quick Preset Buttons -->
    <div class="presets">
        <span style="font-size: 13px; color: #64748b;">หรือลองภาพตัวอย่าง:</span>
        <button class="preset-btn" onclick="loadSamplePreset('03_butterfly.png')">🦋 ลายปีกผีเสื้อ (Butterfly)</button>
        <button class="preset-btn" onclick="loadSamplePreset('10_comic.png')">🎨 ลายเส้นการ์ตูน (Comic)</button>
        <button class="preset-btn" onclick="loadSamplePreset('06_baboon.png')">🐒 ขนสัตว์ความถี่สูง (Baboon)</button>
        <button class="preset-btn" onclick="loadSamplePreset('01_baby.png')">👶 ใบหน้าทารก (Baby)</button>
        <button class="preset-btn" onclick="loadSamplePreset('20_foliage.png')">🌿 กิ่งไม้/ใบไม้ (Foliage)</button>
    </div>

    <!-- Loader -->
    <div id="loader">
        <div class="spinner"></div>
        <div style="font-size: 15px; color: #38bdf8; font-weight: 600;">กำลังรัน Inference ขยายภาพ 4× ด้วยทุกโมเดล และคำนวณ Metrics...</div>
        <div style="font-size: 13px; color: #64748b; margin-top: 4px;">(Bicubic, ESRGAN, BSRGAN, Real-ESRNet, Ours 52k, Real-ESRGAN Official)</div>
    </div>

    <!-- Results -->
    <div id="resultSection">
        <h2 style="font-size: 20px; color: #f8fafc; margin: 30px 0 16px 0;">1. ภาพผลลัพธ์การขยายขนาด 4× (Hover เพื่อ Zoom ตรวจความคมชัด)</h2>
        <div class="gallery-grid" id="galleryGrid"></div>

        <!-- Summary Table -->
        <div class="table-card">
            <div class="table-title">2. ตารางวัดผลประสิทธิภาพและคุณภาพของภาพที่อัปโหลด (Metrics Summary)</div>
            <table>
                <thead>
                    <tr>
                        <th>โมเดล</th>
                        <th>NIQE <span class="arrow-down">↓</span> (ยิ่งต่ำยิ่งเป็นธรรมชาติ)</th>
                        <th>MUSIQ <span class="arrow-up">↑</span> (คะแนนความสวยงาม)</th>
                        <th>Resolution</th>
                        <th>Inference Time</th>
                    </tr>
                </thead>
                <tbody id="metricsTableBody"></tbody>
            </table>
            <div style="font-size: 13px; color: #64748b; margin-top: 14px;">
                * <strong>NIQE:</strong> ยิ่งต่ำยิ่งดี (วัดความเป็นธรรมชาติ ไม่หลอกตา) | <strong>MUSIQ:</strong> ยิ่งสูงยิ่งดี (วัดความสวยงามคมชัดโดยใช้ Transformer-based IQA)
            </div>
        </div>
    </div>
</div>

<script>
    const dropArea = document.getElementById('dropArea');

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => { e.preventDefault(); dropArea.classList.add('dragover'); }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => { e.preventDefault(); dropArea.classList.remove('dragover'); }, false);
    });
    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) handleFile(dt.files[0]);
    });

    function handleFile(file) {
        if (!file) return;
        const formData = new FormData();
        formData.append('image', file);
        processImage(formData);
    }

    function loadSamplePreset(filename) {
        document.getElementById('loader').style.display = 'block';
        document.getElementById('resultSection').style.display = 'none';

        fetch('/api/preset?name=' + filename)
            .then(res => res.json())
            .then(data => renderResults(data))
            .catch(err => { alert('Error: ' + err); document.getElementById('loader').style.display = 'none'; });
    }

    function processImage(formData) {
        document.getElementById('loader').style.display = 'block';
        document.getElementById('resultSection').style.display = 'none';

        fetch('/api/upscale', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => renderResults(data))
            .catch(err => { alert('Error processing image: ' + err); document.getElementById('loader').style.display = 'none'; });
    }

    function renderResults(data) {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('resultSection').style.display = 'block';

        const galleryGrid = document.getElementById('galleryGrid');
        const tableBody = document.getElementById('metricsTableBody');
        galleryGrid.innerHTML = '';
        tableBody.innerHTML = '';

        // Find best scores
        let minNiqe = Math.min(...data.results.map(r => r.niqe));
        let maxMusiq = Math.max(...data.results.map(r => r.musiq));

        // Render Cards
        // 1. Original Input Card
        const lrCard = document.createElement('div');
        lrCard.className = 'model-card';
        lrCard.innerHTML = `
            <div class="card-header">Input (ภาพต้นฉบับ)</div>
            <div class="card-img-wrap"><img src="${data.input_image}" alt="Input"></div>
            <div class="card-footer"><span>Size: ${data.input_shape}</span></div>
        `;
        galleryGrid.appendChild(lrCard);

        // 2. Models Cards & Table Rows
        data.results.forEach(res => {
            const isTeam = res.model.includes('Ours');
            const card = document.createElement('div');
            card.className = 'model-card ' + (isTeam ? 'highlight-team' : '');
            card.innerHTML = `
                <div class="card-header">${res.model} ${isTeam ? '<span class="badge-team">โมเดลของทีม</span>' : ''}</div>
                <div class="card-img-wrap"><img src="${res.image}" alt="${res.model}"></div>
                <div class="card-footer">
                    <span>NIQE: <strong>${res.niqe.toFixed(2)}</strong></span>
                    <span>MUSIQ: <strong>${res.musiq.toFixed(1)}</strong></span>
                </div>
            `;
            galleryGrid.appendChild(card);

            // Table Row
            const row = document.createElement('tr');
            if (isTeam) row.className = 'tr-team';
            row.innerHTML = `
                <td><strong>${res.model}</strong></td>
                <td class="${res.niqe === minNiqe ? 'val-best' : ''}">${res.niqe.toFixed(3)} ${res.niqe === minNiqe ? '★ ดีที่สุด' : ''}</td>
                <td class="${res.musiq === maxMusiq ? 'val-best' : ''}">${res.musiq.toFixed(2)} ${res.musiq === maxMusiq ? '★ ดีที่สุด' : ''}</td>
                <td>${res.output_shape}</td>
                <td>${res.time_sec.toFixed(2)} s</td>
            `;
            tableBody.appendChild(row);
        });

        document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
    }
</script>

</body>
</html>
"""

def process_cv2_image(img):
    h, w = img.shape[:2]
    # Resize input if excessively large for fast demo
    max_d = 512
    if max(h, w) > max_d:
        scale = max_d / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Convert to base64 for input
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    input_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    results = []

    # 1. Bicubic
    t0 = time.time()
    sr_bicubic = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    t_bicubic = time.time() - t0
    cv2.imwrite('_tmp_bicubic.png', sr_bicubic)
    with torch.no_grad():
        niqe_b = nr_metrics['niqe']('_tmp_bicubic.png').item()
        musiq_b = nr_metrics['musiq']('_tmp_bicubic.png').item()
    _, b_buf = cv2.imencode('.jpg', sr_bicubic, [cv2.IMWRITE_JPEG_QUALITY, 85])
    results.append({
        'model': 'Bicubic (Baseline)',
        'image': "data:image/jpeg;base64," + base64.b64encode(b_buf).decode('utf-8'),
        'niqe': round(niqe_b, 3),
        'musiq': round(musiq_b, 2),
        'output_shape': f"{w*4}x{h*4}",
        'time_sec': t_bicubic
    })

    # Prepare tensor
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. DL Models
    for m_name, model in models.items():
        t0 = time.time()
        with torch.no_grad():
            output = model(tensor)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            out_bgr = (output.transpose(1, 2, 0)[:, :, [2, 1, 0]] * 255.0).round().astype('uint8')
        t_model = time.time() - t0

        cv2.imwrite('_tmp_out.png', out_bgr)
        with torch.no_grad():
            niqe_val = nr_metrics['niqe']('_tmp_out.png').item()
            musiq_val = nr_metrics['musiq']('_tmp_out.png').item()

        _, m_buf = cv2.imencode('.jpg', out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append({
            'model': m_name,
            'image': "data:image/jpeg;base64," + base64.b64encode(m_buf).decode('utf-8'),
            'niqe': round(niqe_val, 3),
            'musiq': round(musiq_val, 2),
            'output_shape': f"{w*4}x{h*4}",
            'time_sec': t_model
        })

    return {
        'input_image': input_b64,
        'input_shape': f"{w}x{h}",
        'results': results
    }

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upscale', methods=['POST'])
def api_upscale():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    file = request.files['image']
    in_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400
    data = process_cv2_image(img)
    return jsonify(data)

@app.route('/api/preset', methods=['GET'])
def api_preset():
    fname = request.args.get('name', '03_butterfly.png')
    path = os.path.join('datasets/eval_20/LR', fname)
    if not os.path.exists(path):
        return jsonify({'error': 'Preset not found'}), 404
    img = cv2.imread(path)
    data = process_cv2_image(img)
    return jsonify(data)

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" Super-Resolution Comparison Web App is Ready!")
    print(" Open URL: http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

