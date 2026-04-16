# Notebook Execution Guide

## AI in Health Informatics: Kidney Failure Prediction

### `ckd_thesis_notebook_v3.ipynb`

---

> **Version:** v3 (Corrected Pipeline)  
> **Last Updated:** 2025  
> **Python Version Required:** 3.8 or higher  
> **Estimated Runtime:** ~8–15 minutes (CPU) | ~3–5 minutes (GPU)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Placement](#3-dataset-placement)
4. [Notebook Section Walkthrough](#4-notebook-section-walkthrough)
5. [Expected Outputs at Each Stage](#5-expected-outputs-at-each-stage)
6. [Common Errors and Fixes](#6-common-errors-and-fixes)
7. [Running on Google Colab](#7-running-on-google-colab)
8. [Reproducibility Notes](#8-reproducibility-notes)

---

## 1. Prerequisites

Before opening the notebook, ensure you have the following installed:

### Required Python Libraries

| Library           | Purpose                              | Minimum Version |
| ----------------- | ------------------------------------ | --------------- |
| `pandas`          | Data loading and manipulation        | 1.3+            |
| `numpy`           | Numerical operations                 | 1.21+           |
| `matplotlib`      | Static visualisation                 | 3.4+            |
| `seaborn`         | Statistical visualisation            | 0.11+           |
| `scikit-learn`    | ML models, preprocessing, evaluation | 1.0+            |
| `xgboost`         | Gradient boosting classifier         | 1.5+            |
| `torch` (PyTorch) | LSTM and GRU deep learning models    | 1.11+           |
| `shap`            | Model interpretability (SHAP values) | 0.41+           |
| `ipywidgets`      | Notebook display utilities           | 7.0+            |

### Install All at Once

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch shap ipywidgets
```

If you are using **Anaconda**:

```bash
conda install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost shap
conda install pytorch -c pytorch
```

---

## 2. Environment Setup

### Option A — Local Jupyter

```bash
# Install Jupyter if not already installed
pip install notebook

# Launch Jupyter
jupyter notebook

# Navigate to the notebook file in the browser that opens
```

### Option B — JupyterLab (recommended for a cleaner interface)

```bash
pip install jupyterlab
jupyter lab
```

### Option C — VS Code

1. Install the **Python** and **Jupyter** extensions from the VS Code marketplace
2. Open the `.ipynb` file directly in VS Code
3. Select your Python interpreter (top-right of the notebook)
4. Run cells with `Shift + Enter`

---

## 3. Dataset Placement

The notebook loads the CSV with:

```python
df_raw = pd.read_csv('kidney_disease.csv')
```

**The CSV file must be in the same directory as the notebook.** Your folder structure should look like:

```
your_project_folder/
├── ckd_thesis_notebook_v3.ipynb   ← the notebook
└── kidney_disease.csv              ← the dataset
```

If the CSV is located elsewhere, update **Cell [04]** (Section 3 — Load Dataset):

```python
# Example: CSV is in a subfolder called 'data'
df_raw = pd.read_csv('data/kidney_disease.csv')

# Example: absolute path on Windows
df_raw = pd.read_csv(r'C:\Users\YourName\thesis\kidney_disease.csv')

# Example: absolute path on macOS/Linux
df_raw = pd.read_csv('/home/yourname/thesis/kidney_disease.csv')
```

---

## 4. Notebook Section Walkthrough

The notebook contains **77 cells** (40 code + 37 markdown). Run them **strictly in order from top to bottom**. The sections are as follows:

---

### Section 1 — Introduction _(Cell 00, Markdown)_

**What it does:** Provides academic background on CKD, states the research objectives, and contains a critical note explaining the preprocessing bug found in earlier versions.

**Action required:** Read only. No code to run.

> **Important:** The bug description in this section (LabelEncoder NaN-as-class leakage) is documented for academic transparency. The v3 notebook has already fixed this issue. You do not need to do anything — just read and understand it for your thesis write-up.

---

### Section 2 — Import Libraries _(Cell 02, Code)_

**What it does:** Imports all required Python libraries and sets the global random seed (`SEED = 42`) for reproducibility.

**How to run:** Click the cell → press `Shift + Enter`.

**Expected output:**

```
All libraries imported successfully.
PyTorch : 2.x.x
Device  : CPU
```

**If you see an ImportError:**

- Run the install command from Section 1 of this guide
- Restart the kernel (`Kernel → Restart`) and re-run this cell

---

### Section 3 — Load Dataset _(Cells 03–07, Code)_

**What it does:** Reads the CSV file and displays:

- Dataset shape (should be `400 rows × 25 columns`)
- First 5 rows (`df_raw.head()`)
- Column data types and null counts (`df_raw.info()`)
- Statistical summary (`df_raw.describe()`)

**Expected output for Cell 04:**

```
Shape   : (400, 25)  (400 patients × 25 attributes)
Columns : ['age', 'bp', 'sg', ..., 'classification']
```

**If you see `FileNotFoundError`:** The CSV is not in the correct location. See Section 3 of this guide.

---

### Section 4 — Data Preprocessing _(Cells 08–17, Code + Markdown)_

This is the **most important section**. It contains the corrected preprocessing pipeline. Run all cells in this order:

#### Cell 09–10 | String Cleaning & Dtype Fixes

**What it does:**

1. Strips leading/trailing whitespace and tab characters from all string columns (fixes dirty values like `' yes'`, `'\tno'` in `dm` and `cad`)
2. Fixes the target label `'ckd\t'` → `'ckd'`
3. Converts `pcv`, `wc`, and `rc` from string dtype to numeric using `pd.to_numeric(..., errors='coerce')`

**Expected output:**

```
Target BEFORE fix: {'ckd': 248, 'notckd': 150, 'ckd\t': 2}
Target AFTER  fix: {'ckd': 250, 'notckd': 150}

Dtypes after correction:
pcv    float64
wc     float64
rc     float64
```

> **Why this matters:** `pcv`, `wc`, and `rc` store numeric values (e.g., packed cell volume = 44%) as plain text in the CSV. If not converted, they get treated as categorical text, completely destroying their clinical meaning.

---

#### Cell 11–13 | Missing Value Analysis

**What it does:** Counts and visualises missing values per feature before any imputation.

**Expected output (key figures):**

- `rc`: ~131 missing (33%)
- `wc`: ~106 missing (26%)
- `pcv`: ~71 missing (18%)
- `rbc`: ~152 missing (38%)

A bar chart is produced showing missing percentage per feature with threshold lines at 20% and 30%.

---

#### Cell 14 | Missingness Pattern Analysis (Critical Fix Documentation)

**What it does:** Prints a table showing what percentage of CKD vs non-CKD patients have missing values in each categorical column. This is the **proof of why the old approach caused leakage**.

**Expected output (key rows):**

```
Feature   Total NaN   NaN in CKD (%)   NaN in notCKD (%)
rbc            152             57.2%               6.0%
pc              65             22.4%               6.0%
```

> This output directly demonstrates that `rbc` missingness is strongly correlated with CKD status. If NaN is encoded as a separate class, the model learns `missing = CKD`.

---

#### Cell 15–16 | Correct Encoding: Mode Imputation → Label Encoding

**What it does (the critical fix):**

1. For each categorical column, fills NaN with the **mode** (most frequent real value)
2. Only then applies `LabelEncoder` — so only genuine category values (e.g., `'normal'`, `'abnormal'`) receive integer codes
3. Encodes the target: `'ckd' → 1`, `'notckd' → 0`

**Expected output:**

```
NaN in categorical columns after correct encoding:
{'rbc': 0, 'pc': 0, 'pcc': 0, 'ba': 0, 'htn': 0, ...}

NaN in numerical columns (will be imputed AFTER split):
{'age': 9, 'bp': 12, 'sg': 47, ...}
```

> ✅ **Verification:** All categorical columns should show `0` NaN. Numerical columns will still show NaN — this is correct, as they are handled after the split.

---

#### Cell 17 | Train-Test Split → Impute → Scale

**What it does (correct order):**

1. Splits data: 80% train (320 samples), 20% test (80 samples), stratified by class
2. Fits `SimpleImputer(strategy='median')` **on training data only**, then transforms both sets
3. Fits `StandardScaler` **on imputed training data only**, then transforms both sets

**Expected output:**

```
Training set : (320, 24)
Test set     : (80, 24)
Train class balance : {1: 200, 0: 120}
Test  class balance : {1: 50, 0: 30}
```

> **Why this order matters:** Fitting the imputer or scaler on the full dataset (including the test set) leaks test-set statistics into training. With `rc` having 33% missing values, this would have an especially large effect.

---

### Section 5 — Exploratory Data Analysis _(Cells 18–30, Code + Markdown)_

This section produces **6 sets of visualisations**. All are read-only — no inputs required.

| Cell | Plot Type                       | Key Insight                                              |
| ---- | ------------------------------- | -------------------------------------------------------- |
| 20   | Pie + Bar (class balance)       | 62.5% CKD, 37.5% not-CKD                                 |
| 22   | 4×4 histogram grid              | `sc`, `bu` are right-skewed; `hemo` is bimodal           |
| 24   | 2×4 box plots (by class)        | `hemo` and `sg` show near-zero overlap between classes   |
| 26   | 2×5 count plots (categoricals)  | `htn` and `dm` strongly associated with CKD              |
| 28   | Correlation heatmap             | `hemo`, `pcv`, `rc` are co-correlated (same blood count) |
| 30   | Horizontal bar (univariate AUC) | `hemo` AUC = 0.97, `sc` AUC = 0.92 before any model      |

Run all cells in order. Each produces a matplotlib figure inline.

---

### Section 6 — Classical Machine Learning Models _(Cells 31–44, Code + Markdown)_

**What it does:** Trains 6 models sequentially. For each model, **two evaluations** are performed:

1. `evaluate_holdout(...)` — trains on `X_train_sc`, predicts on `X_test_sc`, prints metrics + confusion matrix
2. `evaluate_cv(...)` — runs 10-fold stratified cross-validation using a **full pipeline** (impute → scale → model) to avoid any leakage even within CV folds

**Models trained:**

| Cell | Model               | Key Hyperparameters                         |
| ---- | ------------------- | ------------------------------------------- |
| 34   | Logistic Regression | `max_iter=1000`, L2 regularisation          |
| 36   | Decision Tree       | Default (no depth limit)                    |
| 38   | Random Forest       | `n_estimators=100`                          |
| 40   | SVM (RBF)           | `kernel='rbf'`, `probability=True`          |
| 42   | Gradient Boosting   | `n_estimators=100`                          |
| 44   | XGBoost             | `n_estimators=100`, `eval_metric='logloss'` |

**Expected outputs per model:**

- Printed metric table (Accuracy, Precision, Recall, F1)
- Full `classification_report` (per-class precision, recall, F1, support)
- Confusion matrix plot (2×2, blue colour map)
- One-line CV result printed inline

> **Runtime note:** SVM and Gradient Boosting are the slowest models here (~30–60 seconds each on CPU for the CV step). XGBoost is faster despite being a boosting method.

---

### Section 7 — Model Optimization _(Cells 45–47, Code)_

**What it does:** Applies `GridSearchCV` with 5-fold CV to tune hyperparameters for:

- **Random Forest:** `n_estimators` ∈ {50, 100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5}
- **XGBoost:** `n_estimators` ∈ {50, 100, 200}, `max_depth` ∈ {3, 6, 9}, `learning_rate` ∈ {0.01, 0.1, 0.3}

The grid search runs inside a **full pipeline** (impute → scale → model) so no leakage occurs during the inner CV folds.

**Expected output:**

```
Best params : {'clf__max_depth': None, 'clf__min_samples_split': 2, 'clf__n_estimators': 200}
Best CV F1  : 0.9941
```

> **Runtime note:** Each grid search evaluates 18 parameter combinations × 5 folds = 90 model fits. This takes approximately **2–4 minutes** per model on CPU. This is the slowest part of the notebook.

---

### Section 8 — Deep Learning Models _(Cells 48–56, Code + Markdown)_

**What it does:** Implements and trains LSTM and GRU classifiers using PyTorch.

#### Cell 49 — Data Preparation for PyTorch

Reshapes data to `(N, 1, n_features)` — treating each patient as a single time step — and wraps in `DataLoader` objects.

#### Cell 50 — Training Loop Function

Defines the reusable `train_rnn()` function. Produces two plots per model:

- Training loss curve over epochs
- Validation accuracy curve over epochs

#### Cells 51–53 — LSTM

- Defines `LSTMClassifier`: 2-layer LSTM → Dropout(0.3) → Linear → Sigmoid
- Trains for 60 epochs, batch size 32, Adam optimiser (lr = 0.001)
- Produces training curves, final metrics, and confusion matrix

**Expected LSTM output:**

```
LSTM parameters: 37,185
Epoch   1/60  |  Loss: 0.6xxx  |  Val Acc: 0.xxxx
Epoch  10/60  |  Loss: 0.3xxx  |  Val Acc: 0.9xxx
...
Epoch  60/60  |  Loss: 0.1xxx  |  Val Acc: 0.97xx
```

#### Cells 54–56 — GRU

Same structure as LSTM. GRU has slightly fewer parameters (~28,000) and typically converges faster.

> **Note on GPU:** If `torch.cuda.is_available()` returns `True`, training will run on GPU and complete in seconds instead of minutes. For 60 epochs on a small dataset, CPU is usually sufficient.

---

### Section 9 — Comparison _(Cells 57–62, Code + Markdown)_

**What it does:**

- **Cell 58:** Builds and displays the styled comparison table (all hold-out results, colour-coded green/red)
- **Cell 59:** Produces a grouped bar chart comparing all metrics across all models
- **Cell 61:** Displays the 10-fold CV table with mean ± std (the primary academic metric)
- **Cell 62:** Provides the written result discussion

No inputs required. Run in order.

---

### Section 10 — ROC Curves _(Cells 63–64, Code)_

**What it does:** Plots ROC curves for all 9 models (7 sklearn + LSTM + GRU) on one chart, with AUC values in the legend.

**Expected output:** A 9×9 inch figure with all curves. Ensemble methods typically reach AUC ≈ 0.999–1.000; LSTM/GRU typically reach AUC ≈ 0.98–0.99.

> **Note:** The optimised models (`rf_opt_model`, `xgb_opt_model`) are sklearn `Pipeline` objects that take raw (unscaled) `X_test` as input. The base models take `X_test_sc` (scaled). This is handled correctly in the ROC cell code.

---

### Section 11 — Model Interpretation _(Cells 65–73, Code + Markdown)_

**What it does:**

- **Cell 67:** Side-by-side horizontal bar charts of Random Forest and XGBoost feature importances
- **Cell 69:** SHAP summary dot plot — shows magnitude and direction for all test-set predictions
- **Cell 70:** SHAP bar plot — mean absolute SHAP value per feature (global ranking)
- **Cell 71:** SHAP waterfall plot — decomposes a single prediction for one CKD patient

> **SHAP note:** SHAP is applied to `xgb_clf` (the raw XGBoost model extracted from the optimised pipeline) with `X_test_sc` (scaled test data). This is correct because the pipeline's scaler has already transformed the data before it reaches the classifier step.

---

### Section 12 — Conclusion _(Cell 74, Markdown)_

Read-only summary. No code to run.

---

## 5. Expected Outputs at Each Stage

| Stage            | Expected Outputs                                                  |
| ---------------- | ----------------------------------------------------------------- |
| Library import   | Version printout, no errors                                       |
| Dataset load     | Shape `(400, 25)`, columns list                                   |
| Cleaning         | Target counts corrected; `pcv`/`wc`/`rc` dtype = float64          |
| Encoding         | Zero NaN in categorical cols; NaN still present in numerical cols |
| Split            | Train `(320, 24)`, Test `(80, 24)`                                |
| Each ML model    | Metrics table + confusion matrix plot + CV F1 printed             |
| Grid search      | Best params printed, optimised model confusion matrix             |
| LSTM training    | Loss/accuracy curves, final Acc ≈ 0.96–0.99                       |
| GRU training     | Loss/accuracy curves, final Acc ≈ 0.96–0.99                       |
| Comparison table | Colour-gradient table sorted by F1-Score                          |
| ROC chart        | All 9 curves on one plot, AUC ≥ 0.98 for all models               |
| SHAP             | Three plots: dot summary, bar summary, waterfall                  |

---

## 6. Running on Google Colab

Google Colab provides free GPU access and requires no local installation.

### Steps

**1. Upload both files to Colab:**

- Open [colab.research.google.com](https://colab.research.google.com)
- Click `File → Upload notebook` → select `ckd_thesis_notebook_v3.ipynb`
- In the left panel, click the folder icon → upload `kidney_disease.csv`

**2. Install missing packages (first cell, run once):**

```python
!pip install xgboost shap --quiet
```

**3. Optionally enable GPU:**

- `Runtime → Change runtime type → Hardware accelerator → GPU`
- PyTorch will automatically detect and use it

**4. Run all cells:**

- `Runtime → Run all` — or run section by section with `Shift + Enter`

### Mounting Google Drive (alternative to direct upload)

```python
from google.colab import drive
drive.mount('/content/drive')

# Then update the CSV path in Cell 04:
df_raw = pd.read_csv('/content/drive/MyDrive/your_folder/kidney_disease.csv')
```

---

## 8. Reproducibility Notes

All stochastic components use `random_state=42` or `torch.manual_seed(42)`. This ensures:

| Component              | Seed Set By                                        |
| ---------------------- | -------------------------------------------------- |
| Train/test split       | `train_test_split(..., random_state=42)`           |
| Cross-validation folds | `StratifiedKFold(..., random_state=42)`            |
| Random Forest          | `RandomForestClassifier(..., random_state=42)`     |
| Decision Tree          | `DecisionTreeClassifier(..., random_state=42)`     |
| SVM                    | `SVC(..., random_state=42)`                        |
| Gradient Boosting      | `GradientBoostingClassifier(..., random_state=42)` |
| XGBoost                | `XGBClassifier(..., random_state=42)`              |
| PyTorch (LSTM/GRU)     | `torch.manual_seed(42)` at top of notebook         |
| NumPy operations       | `np.random.seed(42)` at top of notebook            |

**Note on LSTM/GRU:** PyTorch's CUDA operations are not fully deterministic by default even with a seed. Minor variation (±0.5%) in LSTM/GRU results across runs on GPU is expected and does not affect conclusions. Results on CPU are fully reproducible.

---

_Document prepared for thesis submission: "AI in Health Informatics: Kidney Failure Prediction"_
