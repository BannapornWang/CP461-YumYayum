import os
import glob
import cv2
import base64

def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    max_dim = 300
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

images = sorted(glob.glob('datasets/eval_20/LR/*.png'))

rows = []
for p in images:
    fname = os.path.basename(p)
    name_clean = fname.replace('.png', '').replace('_', ' ').title()
    
    lr_b64 = img_to_b64(os.path.join('datasets/eval_20/LR', fname))
    bicubic_b64 = img_to_b64(os.path.join('results/Albums/Bicubic', fname))
    esrgan_b64 = img_to_b64(os.path.join('results/Albums/ESRGAN', fname))
    bsrgan_b64 = img_to_b64(os.path.join('results/Albums/BSRGAN', fname))
    real_esrnet_b64 = img_to_b64(os.path.join('results/Albums/Real-ESRNet', fname))
    ours_b64 = img_to_b64(os.path.join('results/Albums/Ours', fname))
    realesrgan_b64 = img_to_b64(os.path.join('results/Albums/RealESRGAN', fname))
    gt_b64 = img_to_b64(os.path.join('datasets/eval_20/GT', fname))

    rows.append(f"""
    <tr>
        <td class="col-name"><strong>{name_clean}</strong></td>
        <td>
            <div class="thumb-container">
                <img src="{lr_b64}" alt="LR" loading="lazy">
                <span class="badge badge-lr">LR</span>
            </div>
        </td>
        <td>
            <div class="thumb-container">
                <img src="{bicubic_b64}" alt="Bicubic" loading="lazy">
                <span class="badge badge-sub">Bicubic</span>
            </div>
        </td>
        <td>
            <div class="thumb-container">
                <img src="{esrgan_b64}" alt="ESRGAN" loading="lazy">
                <span class="badge badge-sub">ESRGAN</span>
            </div>
        </td>
        <td>
            <div class="thumb-container">
                <img src="{bsrgan_b64}" alt="BSRGAN" loading="lazy">
                <span class="badge badge-sub">BSRGAN</span>
            </div>
        </td>
        <td>
            <div class="thumb-container">
                <img src="{real_esrnet_b64}" alt="Real-ESRNet" loading="lazy">
                <span class="badge badge-sub">Real-ESRNet</span>
            </div>
        </td>
        <td class="highlight-ours-td">
            <div class="thumb-container">
                <img src="{ours_b64}" alt="Ours" loading="lazy">
                <span class="badge badge-ours">Ours (Team 52k)</span>
            </div>
        </td>
        <td>
            <div class="thumb-container">
                <img src="{realesrgan_b64}" alt="RealESRGAN" loading="lazy">
                <span class="badge badge-realesrgan">RealESRGAN (Official)</span>
            </div>
        </td>
        <td>
            <div class="thumb-container">
                <img src="{gt_b64}" alt="Ground Truth" loading="lazy">
                <span class="badge badge-gt">GT</span>
            </div>
        </td>
    </tr>
    """)

html_table = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ตารางเปรียบเทียบภาพผลลัพธ์ครบทุกโมเดล (20 ภาพ) - Super-Resolution Gallery</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            font-family: 'Sarabun', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        body {{
            background-color: #0b0f19;
            color: #f1f5f9;
            margin: 0;
            padding: 30px 15px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            width: 100%;
            max-width: 1850px;
        }}
        .header {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 24px;
            text-align: center;
        }}
        h1 {{
            font-size: 24px;
            color: #38bdf8;
            margin: 0 0 8px 0;
        }}
        p.desc {{
            color: #94a3b8;
            font-size: 14px;
            margin: 0 0 16px 0;
        }}
        .summary-banner {{
            display: inline-flex;
            gap: 20px;
            background: #1e293b;
            padding: 10px 24px;
            border-radius: 30px;
            font-size: 13.5px;
        }}
        .banner-item strong {{
            color: #60a5fa;
        }}
        .table-card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            overflow-x: auto;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: center;
            min-width: 1500px;
        }}
        th {{
            background: #1f2937;
            color: #e2e8f0;
            padding: 16px 8px;
            font-size: 14px;
            font-weight: 600;
            border-bottom: 2px solid #374151;
        }}
        td {{
            padding: 12px 6px;
            border-bottom: 1px solid #1f2937;
            vertical-align: middle;
        }}
        tr:hover td {{
            background: #172033;
        }}
        .col-name {{
            font-size: 13.5px;
            color: #cbd5e1;
            text-align: left;
            padding-left: 15px;
            white-space: nowrap;
        }}
        .thumb-container {{
            position: relative;
            width: 150px;
            height: 150px;
            margin: 0 auto;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #374151;
            background: #000;
        }}
        .thumb-container img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
            cursor: pointer;
        }}
        .thumb-container img:hover {{
            transform: scale(1.4);
            z-index: 20;
        }}
        .badge {{
            position: absolute;
            bottom: 4px;
            right: 4px;
            font-size: 9.5px;
            font-weight: 600;
            padding: 2px 5px;
            border-radius: 4px;
            backdrop-filter: blur(4px);
        }}
        .badge-lr {{ background: rgba(100, 116, 139, 0.85); color: #fff; }}
        .badge-sub {{ background: rgba(71, 85, 105, 0.85); color: #fff; }}
        .badge-ours {{ background: rgba(37, 99, 235, 0.95); color: #fff; }}
        .badge-realesrgan {{ background: rgba(5, 150, 105, 0.95); color: #fff; }}
        .badge-gt {{ background: rgba(124, 58, 237, 0.95); color: #fff; }}
        .highlight-ours-td {{
            background: rgba(30, 58, 138, 0.25);
            border-left: 1px solid #2563eb;
            border-right: 1px solid #2563eb;
        }}
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>ตารางเปรียบเทียบภาพผลลัพธ์ครบทุกโมเดล (20 ภาพมาตรฐาน ×4)</h1>
        <p class="desc">เปรียบเทียบครบทุกโมเดล: Input (LR) vs Bicubic vs ESRGAN vs BSRGAN vs Real-ESRNet vs โมเดลทีม (Ours) vs RealESRGAN (Official) vs Ground Truth (GT)</p>
        <div class="summary-banner">
            <div class="banner-item">จำนวนโมเดล: <strong>ครบทั้ง 6 โมเดล</strong></div>
            <div class="banner-item">ภาพทั้งหมด: <strong>20 ภาพตัวอย่าง (Set5, Set14, Foliage)</strong></div>
            <div class="banner-item">การทำงาน: <strong>เอาเมาส์ชี้ที่ภาพเพื่อ Zoom ตรวจรายละเอียด</strong></div>
        </div>
    </div>

    <div class="table-card">
        <table>
            <thead>
                <tr>
                    <th style="width: 130px; text-align: left; padding-left: 15px;">ชื่อภาพ</th>
                    <th>LR</th>
                    <th>Bicubic</th>
                    <th>ESRGAN</th>
                    <th>BSRGAN</th>
                    <th>Real-ESRNet</th>
                    <th style="background: #1e3a8a; color: #93c5fd;">Ours (โมเดลทีม 52k)</th>
                    <th>RealESRGAN (Official)</th>
                    <th>Ground Truth (GT)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
</div>

</body>
</html>
"""

with open('model_comparison_20_table.html', 'w', encoding='utf-8') as f:
    f.write(html_table)

print("Updated model_comparison_20_table.html with all 6 models successfully!")
