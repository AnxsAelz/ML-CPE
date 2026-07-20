# LAB 1: Regression – Age Prediction from Facial Images

โปรเจกต์นี้เป็นส่วนหนึ่งของใบงานที่ 3 (Regression & Classification) โดย LAB 1 มุ่งเน้นการทำนาย **อายุ (Age)** จากภาพใบหน้า ด้วยเทคนิค Linear Regression ในสามระดับความซับซ้อน พร้อมประยุกต์ใช้ Principal Component Analysis (PCA) เพื่อลดจำนวนคุณลักษณะของภาพ

## 🎯 วัตถุประสงค์

- เข้าใจหลักการของ Regression ในการทำนายค่าต่อเนื่อง (Continuous Value)
- เปรียบเทียบ Simple Linear Regression และ Multiple Linear Regression
- ประยุกต์ใช้ PCA เพื่อลดมิติของข้อมูลภาพและเพิ่มประสิทธิภาพการเรียนรู้
- ฝึกสร้าง ฝึกสอน ทดสอบ และประเมินผลแบบจำลองด้วย Python และ scikit-learn

## 📊 Dataset

**[Age, Gender and Ethnicity (Face Data) CSV](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv)**

| รายละเอียด | ค่า |
|---|---|
| จำนวนภาพ | 23,705 ภาพ |
| ขนาดภาพ | 48 × 48 พิกเซล (grayscale) |
| คอลัมน์ | `age`, `ethnicity`, `gender`, `img_name`, `pixels` |
| รูปแบบ pixels | string ของค่าความเข้มพิกเซล 0–255 คั่นด้วยช่องว่าง (2,304 ค่า/ภาพ) |

> ไฟล์ `age_gender.csv` ไม่ได้แนบมาในโปรเจกต์นี้เนื่องจากมีขนาดใหญ่ (~200 MB) กรุณาดาวน์โหลดจาก Kaggle ด้วยตนเองแล้ววางไว้ในโฟลเดอร์เดียวกับโน้ตบุ๊ก

## 📁 โครงสร้างไฟล์

```
├── LAB1_Regression_Age_Prediction.ipynb            # โน้ตบุ๊กต้นฉบับ (ยังไม่รัน)
├── LAB1_Regression_Age_Prediction_executed.ipynb   # โน้ตบุ๊กที่รันแล้ว พร้อม output/กราฟ
├── age_gender.csv                                  # ดาวน์โหลดเองจาก Kaggle (ไม่รวมในนี้)
└── README.md
```

## ⚙️ การติดตั้ง

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## ▶️ วิธีรัน

1. ดาวน์โหลด `age_gender.csv` จาก Kaggle แล้ววางไว้โฟลเดอร์เดียวกับโน้ตบุ๊ก
   (หรือแก้ตัวแปร `DATA_PATH` ในเซลล์ที่ 2 ของโน้ตบุ๊กให้ตรงกับตำแหน่งไฟล์)
2. เปิดและรัน `LAB1_Regression_Age_Prediction.ipynb` ตามลำดับเซลล์

```bash
jupyter notebook LAB1_Regression_Age_Prediction.ipynb
```

## 🔬 วิธีการ (Methodology)

| ขั้นตอน | รายละเอียด |
|---|---|
| **1. Data Preparation** | แปลงคอลัมน์ `pixels` (string) เป็นภาพ numpy array 48×48 และ normalize เป็นช่วง 0–1 |
| **2. Simple Linear Regression** | ใช้ 1 feature (ค่าความเข้มพิกเซลเฉลี่ยของภาพ) ทำนายอายุ |
| **3. Multiple Linear Regression** | ใช้ 8 features เชิงสถิติของภาพ (mean, std, min, max, ค่าเฉลี่ยราย quadrant) |
| **4. PCA + Linear Regression** | Standardize พิกเซลทั้งภาพ (2,304 มิติ) → ลดมิติด้วย PCA เหลือ 50 components → fit Linear Regression |
| **5. Evaluation** | เปรียบเทียบด้วย MSE, RMSE, MAE, R² และกราฟ Actual vs Predicted |

## 📈 ผลลัพธ์

| โมเดล | RMSE (ปี) | MAE (ปี) | R² |
|---|---|---|---|
| Simple Linear Regression (1 feature) | 19.56 | 15.21 | 0.010 |
| Multiple Linear Regression (8 features) | 19.13 | 14.80 | 0.053 |
| **PCA (50 components) + Linear Regression** | **15.48** | **12.02** | **0.380** |

PCA ที่ใช้ 50 components สามารถอธิบายความแปรปรวนของข้อมูลภาพได้ 87.47% และให้ผลลัพธ์การทำนายอายุที่แม่นยำกว่าโมเดลอีกสองแบบอย่างชัดเจน สะท้อนว่าการใช้ข้อมูลจากภาพทั้งหมด (ผ่านการลดมิติ) ให้ข้อมูลที่เป็นประโยชน์ต่อการทำนายมากกว่าการสร้าง features เชิงสถิติแบบ manual เพียงไม่กี่ตัว

## 💡 สรุปและข้อจำกัด

- Linear Regression เป็นโมเดลพื้นฐานที่มีสมมติฐานว่าความสัมพันธ์ระหว่าง feature และอายุเป็นเชิงเส้น ซึ่งในทางปฏิบัติภาพใบหน้ามีความสัมพันธ์กับอายุแบบไม่เป็นเชิงเส้น (non-linear) สูง จึงได้ค่า R² ไม่สูงมากแม้จะใช้ PCA แล้ว
- ในงานจริงมักใช้โมเดลที่ซับซ้อนกว่า เช่น CNN (Convolutional Neural Network) เพื่อให้ได้ความแม่นยำที่สูงขึ้น
- แล็บนี้มุ่งเน้นให้เข้าใจ **หลักการพื้นฐานของ Regression, การเตรียมข้อมูลภาพ, และการใช้ PCA** เป็นรากฐานก่อนต่อยอด

## 🔜 ขั้นตอนถัดไป

- **LAB 2: Classification** – ทำนายเพศ (Gender) จากภาพใบหน้า ด้วยชุดข้อมูลเดียวกัน
- ประเมินผลด้วย Accuracy, Precision, Recall, F1-score, ROC Curve, AUC
- เผยแพร่ผลงานฉบับสมบูรณ์บน GitHub เพื่อจัดทำ Portfolio

## 🛠️ Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter

## 📄 License

จัดทำเพื่อการศึกษา ภายใต้รายวิชา Machine Learning — ใบงานที่ 3: Regression & Classification
