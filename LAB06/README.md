# Fake Bills Classification with Neural Network

ใบงานที่ 6 — Neural Network และการประยุกต์ใช้งาน: การจำแนกธนบัตรจริง/ปลอมด้วย Multi-Layer Perceptron (MLP)

## Dataset

[Fake Bills](https://www.kaggle.com/datasets/alexandrepetit881234/fake-bills) — 1,500 ธนบัตร (1,000 จริง / 500 ปลอม) พร้อม 6 คุณลักษณะทางเรขาคณิตที่วัดจากภาพธนบัตร:

| Column | Description |
|---|---|
| `is_genuine` | Target: `True` = ธนบัตรจริง, `False` = ธนบัตรปลอม |
| `diagonal` | ความยาวเส้นทแยงมุม |
| `height_left` | ความสูงด้านซ้าย |
| `height_right` | ความสูงด้านขวา |
| `margin_low` | ระยะขอบล่าง (มีค่าว่างบางส่วน — เติมด้วย median) |
| `margin_up` | ระยะขอบบน |
| `length` | ความยาวธนบัตร |

## Project Structure

```
fake-bills-nn/
├── data/
│   └── fake_bills.csv              # dataset (semicolon-separated)
├── fake_bills_neural_network.ipynb # main notebook (fully executed)
├── predictions.csv                 # predictions on the test set
└── README.md
```

## Method

1. **Load & explore** — check shape, missing values, class balance, correlation, feature distributions
2. **Preprocess** — impute missing `margin_low` values with the median; encode `is_genuine` as 0/1
3. **Split** — 80/20 stratified train/test split
4. **Standardize** — `StandardScaler` fit on the training set only
5. **Model** — `MLPClassifier` (scikit-learn) as the Neural Network
6. **Experiment 1 — Epochs**: fixed architecture `(32,)`, compared `max_iter ∈ {5, 10, 25, 50, 100, 200, 300}`
7. **Experiment 2 — Architecture**: fixed epochs (200), compared 1–3 hidden layers with 8–64 neurons
8. **Evaluation** — accuracy, confusion matrix, classification report, ROC/AUC, training loss curve, validation accuracy curve
9. **Predictions** — exported to `predictions.csv`

## Results

### Effect of epochs (architecture fixed at 1 layer × 32 neurons)

| Epochs | Train Acc. | Test Acc. | Final Loss |
|---|---|---|---|
| 5   | 0.758 | 0.757 | 0.516 |
| 10  | 0.930 | 0.923 | 0.367 |
| 25  | 0.982 | 0.983 | 0.154 |
| 50  | 0.990 | 0.983 | 0.067 |
| 100 | 0.991 | 0.990 | 0.035 |
| 200 | 0.992 | 0.990 | 0.027 |
| 300 | 0.992 | 0.990 | 0.027 |

Accuracy improves quickly up to ~100 epochs, then plateaus — additional epochs beyond that give negligible improvement (the optimizer converges around iteration 156).

### Effect of architecture (200 epochs)

| Architecture | Hidden Layers | Total Neurons | Train Acc. | Test Acc. |
|---|---|---|---|---|
| **1 layer – 8 neurons**   | 1 | 8   | 0.991 | **0.993** |
| 1 layer – 32 neurons      | 1 | 32  | 0.992 | 0.990 |
| 1 layer – 64 neurons      | 1 | 64  | 0.993 | 0.987 |
| 2 layers – (32, 16)       | 2 | 48  | 0.994 | 0.987 |
| 2 layers – (64, 32)       | 2 | 96  | 0.999 | 0.983 |
| 3 layers – (64, 32, 16)   | 3 | 112 | 0.999 | 0.973 |

The smallest network (1 hidden layer, 8 neurons) generalized best on the test set. Larger/deeper networks fit the training data almost perfectly but slightly **overfit**, since this dataset is small (1,500 rows) and the classes are already close to linearly separable.

### Best model

- Final test accuracy ≈ **99%**, with a strong ROC-AUC and near-perfect confusion matrix (only a handful of misclassifications out of 300 test samples).

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook fake_bills_neural_network.ipynb
```

Run all cells top-to-bottom — the dataset is already included under `data/fake_bills.csv`.
