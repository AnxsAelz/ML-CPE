# ใบงานที่ 4 — KNN: Fake Bills Classification

LEB 1: การใช้ K-Nearest Neighbors (KNN) จำแนกธนบัตรจริง/ปลอม จากชุดข้อมูล [Fake Bills (Kaggle)](https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills)

## ไฟล์ในชุดนี้
- `knn_fake_bills.ipynb` — Jupyter Notebook ฉบับสมบูรณ์ (รันผ่านแล้ว มีผลลัพธ์และกราฟครบ)
- `fake_bills.csv` — ชุดข้อมูลที่ใช้ (1,500 แถว × 7 คอลัมน์)

## เนื้อหาใน Notebook
1. โหลดและสำรวจข้อมูล (EDA) — สัดส่วนคลาส, missing values, boxplot, correlation heatmap
2. Preprocessing — เติมค่าว่างของ `margin_low` ด้วยค่ามัธยฐาน
3. แบ่งข้อมูล Train/Test (80/20, stratified)
4. Standardize ฟีเจอร์ด้วย `StandardScaler` (fit บน train เท่านั้น)
5. ฝึกโมเดล KNN ด้วย k = 3, 5, 7 และเปรียบเทียบ Accuracy
6. สำรวจเพิ่มเติมด้วย k = 1–25 เพื่อดูแนวโน้ม
7. Confusion matrix / classification report ของโมเดลที่ดีที่สุด
8. สรุปและอภิปรายผลการทดลอง

## ผลลัพธ์หลัก
| k | Accuracy |
|---|---|
| 3 | 0.9800 |
| 5 | **0.9833** |
| 7 | 0.9833 |

k ที่ดีที่สุด (จากช่วงที่กำหนด) คือ **k = 5** (Accuracy สูงสุดเสมอกับ k=7 แต่ k=5 ให้ผลลัพธ์เดียวกันด้วยความซับซ้อนต่ำกว่า)

## วิธีรัน
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook knn_fake_bills.ipynb
```

## แหล่งข้อมูล
Dataset: [Fake Bills — Kaggle (alexandrepetit881234)](https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills)
