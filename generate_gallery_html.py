import os
import cv2
import base64

def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    # Resize thumbnail slightly for lightweight HTML viewing if too big
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    # Crop representative center region (e.g. 250x250 or downscale slightly)
    max_side = 450
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

test_samples = [
    {
        'id': 'anime_comic.jpg',
        'title': '1. ภาพสไตล์อนิเมะ/การ์ตูน (Anime / Line Art)',
        'desc': 'ภาพต้นฉบับมีรอยแตกของ JPEG Compression และขอบเบลอ ทดสอบการลบรอยแตกและฟื้นฟูเส้นขอบคมชัด'
    },
    {
        'id': 'real_building.jpg',
        'title': '2. ภาพสถาปัตยกรรม/สิ่งก่อสร้าง (Architecture Texture)',
        'desc': 'ภาพถ่ายอาคารจริงที่มีรายละเอียดระเบียงและหน้าต่าง ทดสอบการลบ Artifacts โดยไม่สร้างโครงสร้างหลอกตา'
    },
    {
        'id': 'scenery_ost.png',
        'title': '3. ภาพธรรมชาติและกิ่งไม้ (Fine Foliage / Branches)',
        'desc': 'ภาพกิ่งไม้และใบไม้ความถี่สูง ทดสอบการคงอยู่ของรายละเอียดเส้นแขนงละเอียด'
    },
    {
        'id': 'comic.png',
        'title': '4. ลายเส้นการ์ตูนคลาสสิก (Comic Illustration - Set14)',
        'desc': 'ภาพพิมพ์ลายเส้นขาวดำ/สี ทดสอบการกำจัด Ringing/Overshoot รอบตัวหนังสือและลายเส้น'
    },
    {
        'id': 'butterfly.png',
        'title': '5. ลวดลายปีกผีเสื้อ (Fine Wing Texture - Set5)',
        'desc': 'ภาพมาตรฐานที่มี Texture ซับซ้อน ทดสอบความคมชัดสมจริงเทียบกับธรรมชาติ'
    }
]

models = [
    ('Input (LR)', 'datasets/benchmark_gallery/LR'),
    ('Bicubic (Baseline)', 'results/Gallery/Bicubic'),
    ('ESRGAN', 'results/Gallery/ESRGAN'),
    ('Real-ESRNet', 'results/Gallery/Real-ESRNet'),
    ('Real-ESRGAN (ของทีม - 52k)', 'results/Gallery/Real-ESRGAN (Team 52k)'),
    ('Real-ESRGAN (Official)', 'results/Gallery/Real-ESRGAN (Official)')
]

html_rows = []

for sample in test_samples:
    cards = []
    for m_label, m_dir in models:
        img_p = os.path.join(m_dir, sample['id'])
        b64 = img_to_base64(img_p)
        is_team = 'Real-ESRGAN (ของทีม' in m_label
        is_official = 'Official' in m_label
        tag_class = "badge-team" if is_team else ("badge-official" if is_official else "badge-normal")
        
        cards.append(f"""
        <div class="model-col {'highlight-team-col' if is_team else ''}">
            <div class="col-header"><span class="{tag_class}">{m_label}</span></div>
            <div class="img-wrapper">
                <img src="{b64}" alt="{m_label}" loading="lazy">
            </div>
        </div>
        """)
    
    html_rows.append(f"""
    <div class="sample-section">
        <div class="sample-title">{sample['title']}</div>
        <div class="sample-desc">{sample['desc']}</div>
        <div class="comparison-grid">
            {''.join(cards)}
        </div>
    </div>
    """)

html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Comparison Table - Super-Resolution Models</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            font-family: 'Sarabun', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        body {{
            background-color: #0f172a;
            color: #f1f5f9;
            margin: 0;
            padding: 30px 15px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            width: 100%;
            max-width: 1400px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #1e293b;
            border-radius: 14px;
            border: 1px solid #334155;
        }}
        h1 {{
            font-size: 26px;
            margin: 0 0 10px 0;
            color: #38bdf8;
        }}
        p.subtitle {{
            font-size: 15px;
            color: #94a3b8;
            margin: 0;
        }}
        .sample-section {{
            background: #1e293b;
            border-radius: 14px;
            border: 1px solid #334155;
            padding: 24px;
            margin-bottom: 28px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .sample-title {{
            font-size: 19px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 4px;
        }}
        .sample-desc {{
            font-size: 14px;
            color: #94a3b8;
            margin-bottom: 18px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 14px;
        }}
        .model-col {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .model-col:hover {{
            transform: translateY(-3px);
            border-color: #38bdf8;
        }}
        .highlight-team-col {{
            border: 2px solid #3b82f6 !important;
            background: #172554;
        }}
        .col-header {{
            padding: 10px 8px;
            text-align: center;
            font-size: 13px;
            font-weight: 600;
            border-bottom: 1px solid #334155;
        }}
        .badge-normal {{
            color: #cbd5e1;
        }}
        .badge-team {{
            background: #2563eb;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 12px;
            display: inline-block;
        }}
        .badge-official {{
            background: #059669;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 12px;
            display: inline-block;
        }}
        .img-wrapper {{
            width: 100%;
            height: 220px;
            background: #020617;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .img-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        .img-wrapper img:hover {{
            transform: scale(1.15);
        }}
        @media (max-width: 1100px) {{
            .comparison-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}
        @media (max-width: 700px) {{
            .comparison-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>ตารางเปรียบเทียบผลลัพธ์ภาพจำลอง (Visual Comparison Gallery)</h1>
        <p class="subtitle">เปรียบเทียบผลลัพธ์การขยายภาพ 4× (Super-Resolution) แยกแต่ละโมเดล พร้อมภาพจริงจากชุดข้อมูลอนิเมะ สถาปัตยกรรม ธรรมชาติ และภาพมาตรฐาน</p>
    </div>

    {''.join(html_rows)}

</div>

</body>
</html>
"""

with open('model_comparison_gallery.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Generated model_comparison_gallery.html successfully!")

