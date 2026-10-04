# ใบงานที่ 7: Convolutional Neural Network (CNN) — Fake Bills Dataset

จำแนกธนบัตรจริง/ปลอม (`is_genuine`) จากขนาดทางกายภาพ 6 ค่า ด้วย **1D-CNN** (TensorFlow/Keras)
เปรียบเทียบ **จำนวน Epochs** และ **CNN configuration** (จำนวนชั้น Conv, filters, neurons) ด้วย Accuracy

## โครงสร้างโปรเจกต์
```
lab7_cnn/
├── lab7_cnn_fake_bills.ipynb   # Notebook หลัก (รันแล้ว มี output/กราฟครบ)
├── data/fake_bills.csv         # ชุดข้อมูล (1,500 แถว, คั่นด้วย ;)
├── predictions_test.csv        # ผลทำนายชุด Test (300 แถว)
├── predictions_all.csv         # ผลทำนายทั้งชุด (1,500 แถว) + คอลัมน์ split
├── results_epochs.csv          # ตาราง Accuracy เทียบ epochs
├── results_configs.csv         # ตาราง Accuracy เทียบ configuration
├── figures/                    # กราฟทั้งหมดที่ใช้ใน README
├── requirements.txt
└── README.md
```

## ชุดข้อมูล
| รายการ | รายละเอียด |
|---|---|
| ขนาด | 1,500 แถว, 6 features + 1 label |
| Features | `diagonal`, `height_left`, `height_right`, `margin_low`, `margin_up`, `length` |
| Label | `is_genuine` (True = จริง 1,000, False = ปลอม 500 → อัตราส่วน 2:1) |
| Missing | `margin_low` 37 แถว |

![feature distribution](figures/01_feature_distribution.png)

## ขั้นตอนการทำงาน
1. **แบ่งข้อมูล** Train 960 / Validation 240 / Test 300 (64/16/20, stratified, `random_state=42`)
2. **เติม missing** ด้วย median และ **Standardize** (`StandardScaler`) — fit เฉพาะ train เพื่อไม่ให้เกิด data leakage
3. **จัดรูป input** เป็น `(samples, 6, 1)` — มอง 6 features เป็นลำดับความยาว 6, 1 channel
4. **สร้าง CNN** `Conv1D(kernel=3, same, ReLU) → MaxPool(2)` ซ้ำตามจำนวนชั้น → `Flatten → Dense(ReLU) → Dropout(0.2) → Dense(1, sigmoid)`
   Optimizer Adam (lr=1e-3), loss `binary_crossentropy`, batch size 64
5. **ทดลอง** เทรนใหม่ทุกครั้ง รัน 3 seeds (42, 7, 2026) แล้วเฉลี่ย (test set มีแค่ 300 แถว ค่าจาก seed เดียวแกว่งง่าย)

## ผลที่ 1: เปรียบเทียบจำนวน Epochs
โครงสร้างคงที่: 2 Conv (32→64) + Dense 64

| Epochs | Train Acc | Val Acc | Test Acc |
|---:|---:|---:|---:|
| 5 | 98.54% | 97.50% | 97.44% |
| 10 | 99.37% | 98.06% | 98.33% |
| **20** | 99.48% | 98.33% | **98.67%** |
| 50 | 99.72% | 98.33% | 97.67% |
| 100 | 100.00% | 98.33% | 97.22% |

![accuracy vs epochs](figures/02_accuracy_vs_epochs.png)

- 5 epochs ยังเรียนรู้ไม่เต็มที่ → Test ต่ำสุด
- ประมาณ **20 epochs** ให้ Test สูงสุด
- 50–100 epochs: Train ขึ้นถึง 100% แต่ Validation คงที่และ Test ลดลง → **overfit**

## ผลที่ 2: เปรียบเทียบ CNN Configuration
ทุก config เทรน 50 epochs

| Config | Conv layers | Filters | Dense | Params | Train | Val | Test (± std) |
|---|:-:|---|:-:|---:|---:|---:|---:|
| A | 1 | 16 | 16 | 865 | 99.31% | 97.92% | 98.67% ± 0.27 |
| B | 1 | 32 | 64 | 6,401 | 99.48% | 97.92% | 98.44% ± 0.16 |
| C | 2 | 32-64 | 32 | 8,449 | 99.58% | 98.33% | 98.33% ± 0.27 |
| D | 2 | 32-64 | 128 | 14,785 | 99.83% | 98.33% | 97.89% ± 0.42 |
| E | 3 | 16-32-64 | 64 | 12,065 | 99.76% | 98.61% | 97.78% ± 0.42 |
| F | 3 | 32-64-128 | 128 | 47,681 | 100.00% | 98.33% | 97.89% ± 0.31 |

![accuracy by config](figures/03_accuracy_by_config.png)

- ทุก config อยู่ในช่วง ~97.8–98.7% ต่างกันไม่เกิน 1% (≈ 3 แถวจาก 300) — อยู่ในระดับ noise สรุปไม่ได้ว่าโมเดลใหญ่กว่าดีกว่า
- โมเดลเล็กสุด (A, 865 พารามิเตอร์) ไม่แพ้โมเดลใหญ่ เพราะข้อมูลมีแค่ 6 features และแยกคลาสได้ง่าย

## Training / Validation Accuracy และ Loss
![accuracy curves](figures/04_train_val_accuracy_by_config.png)
![loss curves](figures/05_train_val_loss_by_config.png)

## โมเดลที่เลือกและผลบน Test set
เลือกจาก **Validation accuracy** (ไม่ใช้ test เลือกเพื่อเลี่ยงการเลือกโดยดูข้อสอบ) → **Config E (3 Conv 16-32-64 + Dense 64)**

| | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Fake | 0.96 | 0.97 | 0.97 | 100 |
| Genuine | 0.98 | 0.98 | 0.98 | 200 |
| **Accuracy** | | | **0.9767** | 300 |

![confusion matrix](figures/06_confusion_matrix.png)

ทำนายถูก 293 / 300 แถว ผิด 7 แถว (ปลอมแต่ทายว่าจริง 3, จริงแต่ทายว่าปลอม 4)

| row_id | จริง | ทำนาย | prob_genuine |
|---:|---|---|---:|
| 1121 | Fake | Genuine | 0.9988 |
| 341 | Genuine | Fake | 0.0093 |
| 728 | Genuine | Fake | 0.0003 |
| 1252 | Fake | Genuine | 0.7436 |
| 75 | Genuine | Fake | 0.2483 |
| 1160 | Fake | Genuine | 0.9860 |
| 214 | Genuine | Fake | 0.4004 |

## คำอธิบายไฟล์ Predictions
- `predictions_test.csv`: `row_id`, features 6 ค่า, `actual`, `predicted`, `prob_genuine`, `correct`
- `predictions_all.csv`: เหมือนกัน + `split` (`test` หรือ `train/val`) — Accuracy ทั้งชุด 99.2% แต่รวมข้อมูลที่โมเดลเคยเห็นตอนเทรน จึงใช้รายงานผลจริงไม่ได้ ให้ดูค่าจาก Test เท่านั้น

## ข้อสังเกต / ข้อจำกัด
- ข้อมูลเป็นตารางที่ลำดับ feature ไม่มีความหมายเชิงตำแหน่งเหมือนพิกเซลในรูปภาพ ข้อได้เปรียบของ CNN (จับ local pattern) จึงไม่ชัดเจน ผลใกล้เคียง MLP ในใบงานที่ 6 CNN เหมาะกับรูปภาพ/สัญญาณ/อนุกรมเวลามากกว่า
- Test set 300 แถว → 1 แถวที่ต่าง = 0.33% ความต่างระหว่าง config จึงไม่มีนัยสำคัญทางสถิติ
- ผลอาจต่างเล็กน้อยเมื่อรันบนเครื่อง/เวอร์ชัน TensorFlow ต่างกัน แม้ตั้ง seed แล้ว

## วิธีรัน
```bash
pip install -r requirements.txt
jupyter notebook lab7_cnn_fake_bills.ipynb
```
ใช้เวลาประมาณ 6 นาทีบน CPU (1 core)
