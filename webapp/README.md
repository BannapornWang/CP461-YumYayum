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

### 🌐 การเข้าใช้งาน
เมื่อเซิร์ฟเวอร์เริ่มทำงานแล้ว ให้เปิดเว็บบราวเซอร์ไปที่:
👉 **http://localhost:8000** หรือ **http://127.0.0.1:8000**
