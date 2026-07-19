# ใบงานที่ 2: Data Preprocessing
### UNESCO World Heritage Sites — 50 years (1978–2026)

โปรเจกต์นี้เป็นส่วนหนึ่งของใบงานที่ 2 วิชา Data Preprocessing โดยนำ **UNESCO World Heritage Sites Dataset (1978–2026)** มาใช้ฝึกกระบวนการตรวจสอบ ทำความสะอาด แปลงรูปแบบ และเตรียมข้อมูลให้มีคุณภาพก่อนนำไปใช้พัฒนาแบบจำลอง Machine Learning

**Dataset:** [UNESCO World Heritage Sites - 50 years](https://www.kaggle.com/datasets/darkmatternet/unesco-world-heritage-sites-dataset-19782026) (Kaggle)

---

## 🎯 วัตถุประสงค์

1. เข้าใจหลักการและความสำคัญของ Data Preprocessing รวมถึงขั้นตอนการตรวจสอบ ทำความสะอาด แปลงรูปแบบ และเตรียมข้อมูล
2. ตรวจสอบและวิเคราะห์คุณภาพของข้อมูล — Missing Values, Duplicate Data, Outliers, Inconsistent Data
3. ประยุกต์ใช้ Data Cleaning, Data Transformation, Data Encoding, Feature Scaling
4. เขียนโปรแกรม Data Preprocessing ด้วย Python และไลบรารีที่เกี่ยวข้อง
5. นำเสนอผลงานผ่านแพลตฟอร์มออนไลน์ (GitHub / Kaggle)

---

## 📁 โครงสร้างไฟล์

```
.
├── lab2_data_preprocessing_unesco.ipynb   # โน้ตบุ๊กหลัก (โค้ดทั้งหมด)
├── unesco_world_heritage_sites.csv        # ไฟล์ข้อมูล (ดาวน์โหลดเองจาก Kaggle)
└── README.md
```

---

## 🧩 เนื้อหาในโน้ตบุ๊ก

### LAB1: Dataset Exploration
- Load Dataset
- Display Shape
- Display Data Types
- Display Summary Statistics
- Display Missing Values
- Display Duplicate Records
- Display Class Distribution

### LAB2: Data Visualization
- Histogram (ทุกคอลัมน์ตัวเลข)
- Correlation Heatmap

### Part 3: Data Cleaning
- Missing Value Handling (เติมด้วย median สำหรับตัวเลข / mode สำหรับข้อความ)
- Duplicate Removal
- Incorrect Data Correction (ปีนอกช่วง 1978–2026, พื้นที่ค่าติดลบ, ข้อความไม่สอดคล้องกัน)
- Data Type Conversion
- เปรียบเทียบผลของ Mean vs Median ในการเติมค่าที่หายไป

### Part 4: Feature Engineering
- Label Encoding (คอลัมน์ประเภทมรดกโลก เช่น Cultural / Natural / Mixed)
- One-Hot Encoding (คอลัมน์ภูมิภาค/ประเทศ)

---

## ⚙️ วิธีใช้งาน

### 1. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. ดาวน์โหลดข้อมูล
ดาวน์โหลดไฟล์ CSV จาก Kaggle แล้ววางไว้ในโฟลเดอร์เดียวกับโน้ตบุ๊ก
(หรือถ้ารันบน Kaggle Notebook ให้กด **Add Data** แนบ dataset นี้เข้าไปได้เลย)

### 3. ตั้งชื่อคอลัมน์ (ถ้าจำเป็น)
โน้ตบุ๊กจะ **ตรวจจับชื่อคอลัมน์อัตโนมัติ** (ปี, พื้นที่, ประเภท, ประเทศ, ภูมิภาค) จากคำสำคัญในชื่อคอลัมน์
หากตรวจจับผิดหรือไม่พบ ให้ดูรายชื่อคอลัมน์จริงที่พิมพ์ออกมาในเซลล์ LAB1 แล้วกรอกชื่อคอลัมน์ที่ถูกต้อง
ลงในเซลล์ **`COLUMN OVERRIDE`** ก่อนรันเซลล์ที่เหลือ

### 4. รันโน้ตบุ๊ก
เปิดไฟล์ `lab2_data_preprocessing_unesco.ipynb` ด้วย Jupyter Notebook / JupyterLab / Kaggle Notebook / Google Colab แล้วรันทีละเซลล์ตามลำดับ

---

## 🛠️ เทคโนโลยีที่ใช้

| Library | ใช้สำหรับ |
|---|---|
| pandas / numpy | จัดการและวิเคราะห์ข้อมูล |
| matplotlib / seaborn | สร้างกราฟ Histogram และ Heatmap |
| scikit-learn | Label Encoding, One-Hot Encoding |

---

## 📊 สรุปผลที่ได้

- ระบุและจัดการค่า Missing Values, ข้อมูลซ้ำ, และข้อมูลผิดปกติ (ปีนอกช่วง, พื้นที่ติดลบ) ได้สำเร็จ
- พบว่าการเติมค่าด้วย **Median** เหมาะสมกว่า **Mean** สำหรับคอลัมน์ที่มีการกระจายแบบเบ้ (เช่น พื้นที่ของแหล่งมรดกโลก) เนื่องจากทนทานต่อ outlier มากกว่า
- แปลงข้อมูลหมวดหมู่ให้พร้อมสำหรับโมเดล ML ด้วย Label Encoding และ One-Hot Encoding ตามความเหมาะสมของประเภทข้อมูล (ordinal vs nominal)

---

## 👤 ผู้จัดทำ

Anas Chaimongkol
*ใบงานที่ 2 — Data Preprocessing*

## 📄 License

ข้อมูลใช้เพื่อการศึกษาเท่านั้น ลิขสิทธิ์ข้อมูลเป็นของ UNESCO / เจ้าของ dataset บน Kaggle
