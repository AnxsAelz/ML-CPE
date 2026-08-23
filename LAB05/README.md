# ใบงานที่ 5: Support Vector Machine (SVM) และการประยุกต์ใช้งาน SVM

## LEB: SVM on a Dataset of Your Choice

## รายละเอียดชุดข้อมูล

**Dataset:** [Fake Bills](https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills) (Kaggle)

ชุดข้อมูลค่าการวัดทางเรขาคณิตของธนบัตร จำนวน 1,500 ใบ (ธนบัตรจริง 1,000 ใบ / ธนบัตรปลอม 500 ใบ) ใช้สำหรับงานจำแนกประเภท (binary classification) ว่าธนบัตรใบใดเป็นของจริงหรือของปลอม

| คอลัมน์ | ความหมาย |
|---|---|
| `is_genuine` | เป้าหมาย: True = ธนบัตรจริง, False = ธนบัตรปลอม |
| `diagonal` | ความยาวเส้นทแยงมุมของธนบัตร (มม.) |
| `height_left` | ความสูงด้านซ้ายของธนบัตร (มม.) |
| `height_right` | ความสูงด้านขวาของธนบัตร (มม.) |
| `margin_low` | ระยะขอบด้านล่าง (มม.) — มีค่า missing บางส่วน |
| `margin_up` | ระยะขอบด้านบน (มม.) |
| `length` | ความยาวของธนบัตร (มม.) |

## วิธีใช้งาน

1. ดาวน์โหลดไฟล์ `fake_bills.csv` จากลิงก์ Kaggle ด้านบน
2. วางไฟล์ `fake_bills.csv` ไว้ในโฟลเดอร์เดียวกับไฟล์ `SVM_FakeBills.ipynb`
3. เปิดโน้ตบุ๊กแล้วรันทุกเซลล์ตามลำดับ (Restart & Run All)

> **หมายเหตุ:** ถ้ายังไม่มีไฟล์ `fake_bills.csv` โน้ตบุ๊กจะสร้างชุดข้อมูลจำลอง (synthetic placeholder) ที่มีโครงสร้างเดียวกันขึ้นมาแทนโดยอัตโนมัติ เพื่อให้ทุกเซลล์รันได้ครบโดยไม่มี error — แต่ผลลัพธ์ที่ได้จะไม่ใช่ค่าจริง ต้องดาวน์โหลดไฟล์จริงมาวางแล้วรันใหม่ก่อนส่งงาน

## ไลบรารีที่ใช้

- `pandas`, `numpy` — จัดการข้อมูล
- `matplotlib`, `seaborn` — พล็อตกราฟ
- `scikit-learn` — `train_test_split`, `StandardScaler`, `SimpleImputer`, `SVC`, `PCA`, เมทริกซ์ประเมินผล

ติดตั้งด้วยคำสั่ง:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## โครงสร้างของโน้ตบุ๊ก

1. **Import Libraries**
2. **Load Dataset** — โหลดไฟล์จริง หรือใช้ข้อมูลจำลองสำรอง
3. **Exploratory Data Analysis (EDA)** — `info()`, `describe()`, ตรวจค่า missing, กราฟสัดส่วนคลาส, boxplot รายฟีเจอร์, heatmap ความสัมพันธ์
4. **Data Preprocessing** — เติมค่า missing (mean imputation), เข้ารหัส target เป็น 0/1, แบ่ง train/test, ปรับมาตรฐานฟีเจอร์ด้วย `StandardScaler`
5. **Train SVM Models** — ฝึกโมเดล 3 kernel: **Linear, Polynomial (degree=3), RBF**
6. **Evaluation** — accuracy, classification report, confusion matrix ของแต่ละ kernel
7. **Compare Kernel Accuracies** — ตารางและกราฟแท่งเปรียบเทียบ
8. **Decision Boundary Visualization** — ลดมิติด้วย PCA เหลือ 2 มิติ เพื่อพล็อตขอบเขตการตัดสินใจของแต่ละ kernel ให้เห็นภาพ
9. **Predictions on New / Unseen Bill Samples** — ทำนายผลด้วย kernel ที่ดีที่สุด ทั้งจาก test set และตัวอย่างธนบัตรที่กำหนดค่าเอง
10. **สรุปผล (Conclusion)**

## ผลลัพธ์ (Output)

- ตาราง/กราฟเปรียบเทียบค่า accuracy ของ SVM ทั้ง 3 kernel
- Confusion matrix และ classification report ของแต่ละ kernel
- ตัวอย่างการทำนายผลกับข้อมูลธนบัตรใหม่
