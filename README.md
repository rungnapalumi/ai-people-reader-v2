# ai-people-reader-v2

## Workers (วิดีโอ + รายงานทำงานคู่กันอย่างไร)

บน **Render** โปรเจกต์นี้ใช้ **Worker สองตัวแยก service** (ดู `render.yaml`):

| Service (ชื่อตัวอย่าง) | คำสั่ง | หน้าที่ |
|------------------------|--------|---------|
| `ai-people-reader-v2-video-worker` | `python src/worker.py` | คิว `jobs/pending/` เฉพาะ **dots** / **skeleton** |
| `ai-people-reader-v2-report-worker` | `python src/report_worker.py` | คิว **report** + ส่งเมล / `jobs/email_pending/` |

- **ไม่ได้รันใน process เดียว** — ถ้าเปิดแค่ตัวเดียว อีกคิวจะไม่ถูกประมวลผล (เช่น มีแต่ report worker จะไม่มีใครทำ skeleton/dots)
- **ทั้งสองต้อง Live** และใช้ **AWS bucket / env เดียวกัน** ไม่งั้นจะเห็นคิวว่างผิดฝั่ง
- รันโลคัลพร้อมกัน: เปิด 2 เทอร์มินัล รัน `python src/worker.py` กับ `python src/report_worker.py` คนละอัน

ถ้า Blueprint สร้างมาแค่ service เดียว ให้ไปที่ Render Dashboard → เพิ่ม Worker อีกตัวตาม `render.yaml` หรือ Redeploy จาก blueprint ให้ครบทั้งสองรายการ