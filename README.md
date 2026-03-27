# Readmission-DL — City General Hospital 30-day Readmission Prediction

**Student name:** [Your Name]
**Student ID:** [Your ID]
**Submission date:** March 27, 2026

---

## Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

---

## My model

**Architecture:**
* **Type:** Sequential Deep Neural Network.
* **Layers:** 64 -> 32 -> 16 units with ReLU activation.
* **Regularization:** Batch Normalization and 30% Dropout on the first two layers.
* **Output:** Sigmoid activation for binary probability.

**Key preprocessing decisions:**
* **Outlier Correction:** Capped invalid ages (e.g., 999) at 100 and used median imputation for missing clinical labs.
* **Feature Engineering:** Extracted `year`, `month`, and `day` from admission dates.
* **Normalization:** Applied `log1p` transformations to highly skewed clinical features to improve gradient stability.

**How I handled class imbalance:**
I used **Cost-Sensitive Learning** by applying `class_weight` during training. This assigned significantly higher importance to the minority "readmitted" class (weight ~5.5) compared to the majority class, ensuring the model prioritizes identifying positive cases.

---

## Results on validation set

| Metric | Value |
|--------|-------|
| F1 (minority class) | 0.58 |
| Precision (minority) | 0.44 |
| Recall (minority) | 0.82 |
| Decision threshold used | 0.57 |

---

## How to run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow
```

### 2. Train the model

```bash
python notebooks/solution.ipynb
```

### 3. Run inference on the test set

```bash
python src/predict.py --input data/test.csv --output predictions.csv
```

### 4. Repository Structure

readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb
├── src/
│   └── predict.py
├── DECISIONS.md
├── requirements.txt
└── README.md

---

### Limitations and honest assessment
The model currently prioritizes Recall (0.82), meaning it is excellent at catching potential readmissions, but the lower Precision (0.44) suggests a high rate of false positives. With more time, I would explore methods like XGBoost, Random Forest to provide better clinical explainability for healthcare providers.