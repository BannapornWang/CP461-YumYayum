# 🚀 วิธีการรัน Web Application (YumYayum CP461)

### 1. รันผ่าน Script (ง่ายที่สุด)
ดับเบิ้ลคลิกไฟล์ Script ที่โฟลเดอร์หลักของโปรเจกต์:
- **Windows**: ดับเบิ้ลคลิกที่ไฟล์ `run_app.bat` หรือคลิกขวาที่ `run_app.ps1` แล้วเลือก **Run with PowerShell**
- ระบบจะเปิดเบราว์เซอร์ไปที่หน้าเว็บให้อัตโนมัติ

---

### 2. รันผ่าน Terminal / Command Line
เปิด Terminal หรือ PowerShell ที่ root directory ของโปรเจกต์ (`CP461-YumYayum`) แล้วใช้คำสั่ง:

```bash
# กรณีเรียกตรงผ่าน Python venv
.\.venv312\Scripts\python.exe -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload
```

หรือเปิดใช้งาน environment ก่อน:

```bash
# เปิดใช้งาน virtual environment
.\.venv312\Scripts\Activate.ps1

# เริ่มต้นเซิร์ฟเวอร์
uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload
```

---

### 🌐 หน้าต่างและการเข้าใช้งาน (Web Endpoints & Dashboards)
เมื่อเซิร์ฟเวอร์เริ่มทำงานแล้ว สามารถเปิดเข้าใช้งานได้ที่:

| URL Path | หน้าเว็บ | รายละเอียด |
| :--- | :--- | :--- |
| **`/`** | 🏠 **Main Presentation & Studio** | เว็บไซต์นำเสนอแบบ Interactive (Overview, Dataset, Method, Metrics, Results, Live 4× Demo) |
| **`/benchmark`** | 📊 **Benchmark Results (20 Images)** | ตารางผลการทดสอบประสิทธิภาพเชิงปริมาณบน 20 ภาพมาตรฐาน + Ablation Study |
| **`/comparison-table`** | 📑 **20-Image Comparison Table** | ตารางเปรียบเทียบภาพผลลัพธ์รายภาพทั้ง 20 ชุด (Low-Res, Bicubic, โมเดลทีม, RealESRGAN) |
| **`/gallery`** | 🖼️ **Model Visual Gallery** | แกลเลอรีผลลัพธ์ภาพเปรียบเทียบแบบขยายดูรายละเอียดและพื้นผิว |
| **`/eda`** | 📈 **EDA Dashboard** | แดชบอร์ดวิเคราะห์และสำรวจการกระจายตัวของชุดข้อมูล DIV2K, Flickr2K, OTS Clear และ Degradation |

👉 ลิงก์หลัก: **http://localhost:8000** หรือ **http://127.0.0.1:8000**

