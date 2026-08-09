# ใบงานที่ 3: Regression & Classification – Facial Image Analysis

โปรเจกต์นี้ครอบคลุมใบงานที่ 3 ครบทั้งสามแล็บ โดยใช้ภาพใบหน้าจาก dataset เดียวกันในการ:
- **LAB 1 (Regression):** ทำนาย **อายุ (Age)** ด้วย Linear Regression
- **LAB 2 (Classification):** จำแนก **เพศ (Gender)** ด้วย Logistic Regression พร้อม Decision Boundary Visualization
- **LAB 3 (Model Comparison):** เปรียบเทียบโมเดลทั้งหมดจาก LAB 1–2 อย่างเป็นระบบ (Simple vs Multiple, Train vs Test, Regression vs Classification)

ทั้งสามแล็บประยุกต์ใช้ Principal Component Analysis (PCA) เพื่อลดจำนวนคุณลักษณะของภาพ และเปรียบเทียบผลลัพธ์ระหว่างวิธีที่ใช้ features น้อย/มาก และมี/ไม่มี PCA

## 🎯 วัตถุประสงค์

- เข้าใจหลักการของ Regression (ทำนายค่าต่อเนื่อง) และ Classification (จำแนกประเภท) และความแตกต่างระหว่างสองแนวทาง
- เปรียบเทียบ Simple Linear Regression และ Multiple Linear Regression
- ประยุกต์ใช้ PCA เพื่อลดมิติของข้อมูลภาพและเพิ่มประสิทธิภาพการเรียนรู้
- พัฒนาโมเดล Regression สำหรับทำนายอายุ และโมเดล Classification สำหรับจำแนกเพศ พร้อมเปรียบเทียบผลลัพธ์
- ฝึกสร้าง ฝึกสอน ทดสอบ และประเมินผลแบบจำลองด้วย Python และ scikit-learn โดยใช้ตัวชี้วัดที่เหมาะสม (MSE, RMSE, R², Accuracy, Precision, Recall, F1-score, ROC Curve, AUC)

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
├── LAB1_Regression_Age_Prediction.ipynb                  # LAB 1 ต้นฉบับ (ยังไม่รัน)
├── LAB1_Regression_Age_Prediction_executed.ipynb         # LAB 1 ที่รันแล้ว พร้อม output/กราฟ
├── LAB2_Classification_Gender_Prediction.ipynb           # LAB 2 ต้นฉบับ (ยังไม่รัน)
├── LAB2_Classification_Gender_Prediction_executed.ipynb  # LAB 2 ที่รันแล้ว พร้อม output/กราฟ
├── LAB3_Model_Comparison.ipynb                           # LAB 3 ต้นฉบับ (ยังไม่รัน)
├── LAB3_Model_Comparison_executed.ipynb                  # LAB 3 ที่รันแล้ว พร้อม output/กราฟ
├── lab1_regression_results.csv                           # ตารางสรุปผล Regression (สร้างจาก LAB 3)
├── lab2_classification_results.csv                       # ตารางสรุปผล Classification (สร้างจาก LAB 3)
├── age_gender.csv                                        # ดาวน์โหลดเองจาก Kaggle (ไม่รวมในนี้)
└── README.md
```

## ⚙️ การติดตั้ง

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## ▶️ วิธีรัน

1. ดาวน์โหลด `age_gender.csv` จาก Kaggle แล้ววางไว้โฟลเดอร์เดียวกับโน้ตบุ๊ก
   (หรือแก้ตัวแปร `DATA_PATH` ในเซลล์ที่ 2 ของแต่ละโน้ตบุ๊กให้ตรงกับตำแหน่งไฟล์)
2. เปิดและรันโน้ตบุ๊กตามลำดับเซลล์

```bash
jupyter notebook LAB1_Regression_Age_Prediction.ipynb
jupyter notebook LAB2_Classification_Gender_Prediction.ipynb
jupyter notebook LAB3_Model_Comparison.ipynb
```

## 🔬 วิธีการ (Methodology)

### LAB 1: Regression (Age Prediction)

| ขั้นตอน | รายละเอียด |
|---|---|
| **1. Data Preparation** | แปลงคอลัมน์ `pixels` (string) เป็นภาพ numpy array 48×48 และ normalize เป็นช่วง 0–1 |
| **2. Simple Linear Regression** | ใช้ 1 feature (ค่าความเข้มพิกเซลเฉลี่ยของภาพ) ทำนายอายุ |
| **3. Multiple Linear Regression** | ใช้ 8 features เชิงสถิติของภาพ (mean, std, min, max, ค่าเฉลี่ยราย quadrant) |
| **4. PCA + Linear Regression** | Standardize พิกเซลทั้งภาพ (2,304 มิติ) → ลดมิติด้วย PCA เหลือ 50 components → fit Linear Regression |
| **5. Evaluation** | เปรียบเทียบด้วย MSE, RMSE, MAE, R² และกราฟ Actual vs Predicted |

### LAB 2: Classification (Gender Prediction)

| ขั้นตอน | รายละเอียด |
|---|---|
| **1. Preparing Classification Data** | ใช้ pipeline เดียวกับ LAB 1 แปลง `pixels` เป็นภาพ + label เพศ (0 = Male, 1 = Female) |
| **2. Logistic Regression (Baseline)** | Standardize พิกเซลทั้งภาพ (2,304 มิติ) → fit Logistic Regression ตรง ๆ |
| **3. Decision Boundary Visualization** | ลดมิติเหลือ 2 PCA components → fit Logistic Regression ใหม่บน 2 มิติ → วาดขอบเขตการตัดสินใจ |
| **4. PCA + Logistic Regression** | ลดมิติด้วย PCA เหลือ 50 components → fit Logistic Regression |
| **5. Evaluation** | Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve, AUC |

### LAB 3: Model Comparison

| ขั้นตอน | รายละเอียด |
|---|---|
| **1. Simple vs Multiple Linear Regression** | เทรนทั้งสองโมเดลบน train/test split เดียวกัน เพื่อเทียบ RMSE/MAE/R² ได้ตรงกัน |
| **2. Training vs Testing Performance** | เทียบ metric บน train set กับ test set ของทุกโมเดล (Regression + Classification) เพื่อตรวจ overfitting/underfitting |
| **3. Regression vs Classification** | เปรียบเทียบเชิงแนวคิด (ประเภท target, ตัวชี้วัด) และเปรียบเทียบผลลัพธ์จริงของทั้งสองงาน |
| **4. Model Performance Metrics** | รวมตารางสรุปผลทุกโมเดล และ export เป็น CSV (`lab1_regression_results.csv`, `lab2_classification_results.csv`) |

## 📈 ผลลัพธ์

### LAB 1: Regression

| โมเดล | RMSE (ปี) | MAE (ปี) | R² |
|---|---|---|---|
| Simple Linear Regression (1 feature) | 19.56 | 15.21 | 0.010 |
| Multiple Linear Regression (8 features) | 19.13 | 14.80 | 0.053 |
| **PCA (50 components) + Linear Regression** | **15.48** | **12.02** | **0.380** |

PCA ที่ใช้ 50 components สามารถอธิบายความแปรปรวนของข้อมูลภาพได้ 87.47% และให้ผลลัพธ์การทำนายอายุที่แม่นยำกว่าโมเดลอีกสองแบบอย่างชัดเจน

### LAB 2: Classification

| โมเดล | Accuracy | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|---|
| Logistic Regression (Raw Pixels - Baseline) | 83.13% | 0.822 | 0.825 | 0.824 | 0.907 |
| PCA (50 components) + Logistic Regression | 82.45% | 0.825 | 0.803 | 0.814 | 0.904 |

ทั้งสองโมเดลจำแนกเพศได้แม่นยำสูง (Accuracy > 82%, AUC > 0.90) โดย PCA ให้ผลลัพธ์ใกล้เคียง baseline แต่ลดจำนวนมิติจาก 2,304 เหลือ 50 ช่วยลดความเสี่ยง overfitting และเวลาฝึกโมเดล

### LAB 3: Model Comparison

- **Simple vs Multiple LR:** เพิ่ม features จาก 1 → 8 ตัว ลด RMSE ลง ~0.43 ปี และเพิ่ม R² ขึ้น ~0.043
- **Train vs Test:** ทุกโมเดลไม่มี overfitting รุนแรง เนื่องจากเป็นโมเดลเชิงเส้นพื้นฐาน แต่ Baseline Logistic Regression (raw pixels) มีช่องว่างระหว่าง train/test มากกว่ารุ่นที่ใช้ PCA เล็กน้อย
- **Regression vs Classification:** R² ของ Age Prediction (PCA+LR) = 0.38 เทียบกับ Accuracy ของ Gender Prediction (PCA+LogReg) = 0.82 — สะท้อนว่าเพศทำนายได้ง่ายกว่าอายุจากภาพใบหน้าอย่างชัดเจน

## 💡 สรุปและข้อจำกัด

- **Regression (LAB 1):** อายุเป็นค่าที่ทำนายจากภาพใบหน้าได้ยากกว่าเพศ เพราะความสัมพันธ์ระหว่างลักษณะภาพกับอายุมีความไม่เป็นเชิงเส้น (non-linear) สูง จึงได้ R² ไม่สูงมาก (~0.38) แม้ใช้ PCA แล้ว
- **Classification (LAB 2):** เพศมีลักษณะเด่นในภาพใบหน้าที่ชัดเจนกว่า (รูปทรงใบหน้า, ผม) ทำให้ Logistic Regression จำแนกได้แม่นยำสูงกว่าอย่างมีนัยสำคัญเมื่อเทียบกับความแม่นยำของ Regression
- ทั้งสองแล็บใช้โมเดลเชิงเส้นพื้นฐาน (Linear/Logistic Regression) ในงานประยุกต์จริงมักใช้ CNN เพื่อความแม่นยำที่สูงขึ้น แต่แล็บนี้มุ่งให้เข้าใจ **หลักการพื้นฐานของ Regression, Classification, การเตรียมข้อมูลภาพ, การใช้ PCA, และตัวชี้วัดการประเมินผลแต่ละประเภท**
- **LAB 3** ยืนยันว่าตัวชี้วัดของ Regression (MSE/RMSE/MAE/R²) และ Classification (Accuracy/Precision/Recall/F1/AUC) ไม่สามารถเทียบกันตรง ๆ ได้ แต่ใช้เปรียบเทียบภายในกลุ่มงานเดียวกันเพื่อเลือกโมเดลที่ดีที่สุดสำหรับแต่ละปัญหาได้

### สรุปโมเดลที่แนะนำ

| งาน | โมเดลที่แนะนำ | เหตุผล |
|---|---|---|
| Age Prediction (Regression) | PCA (50) + Linear Regression | RMSE ต่ำสุด, R² สูงสุดในกลุ่ม Regression |
| Gender Prediction (Classification) | PCA (50) + Logistic Regression | Accuracy/AUC ใกล้เคียง baseline แต่ลดความเสี่ยง overfitting และเวลาฝึกโมเดล |

## 🔜 ขั้นตอนถัดไป

- ทดลองปรับจำนวน PCA components หรือเปลี่ยนโมเดลเป็น Random Forest / SVM / Polynomial Regression เพื่อเปรียบเทียบเพิ่มเติม
- ต่อยอดสู่โมเดล Deep Learning (CNN) สำหรับความแม่นยำที่สูงขึ้นทั้งสองงาน
- เผยแพร่ผลงานฉบับสมบูรณ์ (โน้ตบุ๊กทั้ง 3 แล็บ + README นี้) บน GitHub เพื่อจัดทำ Portfolio

## 🛠️ Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter

## 📄 License

จัดทำเพื่อการศึกษา ภายใต้รายวิชา Machine Learning — ใบงานที่ 3: Regression & Classification (LAB 1 + LAB 2)
