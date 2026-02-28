# ทำไมรายงานไม่เปลี่ยน? — ต้อง Deploy

## สาเหตุ
รายงาน PDF/DOCX ถูกสร้างโดย **Report Worker** ที่รันบน **Render**
- โค้ดที่แก้อยู่ในเครื่องคุณ (local)
- Render ยังใช้โค้ดเก่าอยู่ จนกว่าจะ deploy

## วิธีแก้: Deploy โค้ดไป Render

### 1. Push โค้ดขึ้น Git
```bash
git add .
git commit -m "Fix EN report format to match Thai (bullet, spacing)"
git push origin main
```

### 2. Deploy บน Render
- เปิด [Render Dashboard](https://dashboard.render.com)
- เลือก service **ai-people-reader-v2-report-worker**
- กด **Manual Deploy** → **Deploy latest commit**
- รอ deploy เสร็จ (~2–5 นาที)

### 3. ทดสอบใหม่
- ไปที่ Operational Test
- อัปโหลดวิดีโอใหม่
- เลือก English report (PDF หรือ DOCX)
- ส่งงาน (Submit)
- รอ job เสร็จ แล้วดาวน์โหลดรายงาน

**รายงานที่สร้างหลัง deploy จะใช้รูปแบบใหม่**
