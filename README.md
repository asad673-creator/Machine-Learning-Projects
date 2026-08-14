# ML Projects: Heart Disease Prediction & Car Price Prediction

This repository contains two beginner-friendly machine learning projects built with `pandas`, `scikit-learn`, `seaborn`, and `matplotlib`:

1. **Heart Disease Classification** — predicts whether a patient has heart disease using logistic regression.
2. **Car Price Prediction** — predicts a used car's selling price using linear regression.

---

## 📁 Repository Structure

```
.
├── heart__1_.ipynb        # Heart disease classification notebook
├── heart.csv               # Heart disease dataset
├── car_price_pred.ipynb    # Car price regression notebook
├── Car_Price.csv            # Car price dataset
└── README.md
```

---

## ❤️ 1. Heart Disease Prediction

**Notebook:** `heart__1_.ipynb`
**Dataset:** `heart.csv` (918 rows × 12 columns)

### Features
`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope` → target: `HeartDisease` (0 = no disease, 1 = disease)

### Workflow
- Exploratory data analysis (distributions, class balance, null/duplicate checks)
- Fixed invalid `0` values in `Cholesterol` and `RestingBP` by imputing with the column mean
- Visualized relationships between categorical features (`Sex`, `ChestPainType`, `FastingBS`) and the target
- One-hot encoded categorical variables
- Scaled numeric features with `StandardScaler`
- Train/test split (80/20)
- Trained a **Logistic Regression** model

### Results
| Metric | Score |
|---|---|
| Accuracy | **86.4%** |
| Precision (class 1) | 0.91 |
| Recall (class 1) | 0.85 |
| F1-score (class 1) | 0.88 |

---

## 🚗 2. Car Price Prediction

**Notebook:** `car_price_pred.ipynb`
**Dataset:** `Car_Price.csv` (4,340 rows × 9 columns)

### Features
`Brand`, `Model`, `Year`, `KM_Driven`, `Fuel`, `Seller_Type`, `Transmission`, `Owner` → target: `Selling_Price`

### Workflow
- Removed duplicate rows and checked for nulls
- Scaled `KM_Driven` and `Selling_Price` to thousands (kept as floats to avoid truncation)
- Dropped the high-cardinality `Model` column (1,491 unique values) to prevent one-hot encoding from exploding the feature space
- One-hot encoded remaining categoricals with `drop_first=True` to avoid the dummy variable trap
- Split data **before** scaling to prevent data leakage
- Fit `StandardScaler` on the training set only, then applied it to the test set
- Trained a **Linear Regression** model
- Evaluated with R², MAE, and RMSE, and inspected a residuals-vs-predicted plot

### Results
| Metric | Score |
|---|---|
| R² | 0.5384 |
| MAE | 180.19 (thousands) |
| RMSE | 385.61 (thousands) |

---

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- seaborn, matplotlib
- scikit-learn (`LogisticRegression`, `LinearRegression`, `StandardScaler`, `train_test_split`, metrics)

## ▶️ How to Run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install pandas numpy seaborn matplotlib scikit-learn
jupyter notebook
```

Open either notebook and run all cells in order.

## 📌 Notes
- Both notebooks are exploratory/educational and use simple baseline models (Logistic/Linear Regression) rather than tuned ensembles — a good starting point for further experimentation (e.g. regularization, tree-based models, hyperparameter tuning).
- Feel free to fork and extend with cross-validation, feature selection, or more advanced models.

## 📄 License
Add a license of your choice (e.g. MIT) if you plan to make this repo public.
