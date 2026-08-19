# Machine Learning Projects

A collection of personal machine learning projects and experiments built with Python. Covers data preprocessing, visualization, model training, evaluation, and deployment on real-world datasets, using `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib`.

## Projects

| Project | Task | Model | Result |
|---|---|---|---|
| [Heart Disease Prediction](Projects/Heart-Disease-Prediction) | Binary classification | Logistic Regression | 86.4% accuracy |
| [Car Price Prediction](Projects/Car-Price-Prediction) | Regression | Linear Regression | R² = 0.5384 |

---

## ❤️ Heart Disease Prediction

Predicts whether a patient is at risk of heart disease from clinical features, and is deployed as an interactive **Streamlit web app**.

**Location:** `Projects/Heart-Disease-Prediction/`
**Dataset:** `heart.csv` (918 rows × 12 columns)
**Features:** `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope` → target: `HeartDisease`

**Workflow:**
- EDA (distributions, class balance, null/duplicate checks)
- Imputed invalid `0` values in `Cholesterol` and `RestingBP` with the column mean
- One-hot encoded categorical variables, scaled numeric features with `StandardScaler`
- Trained and compared 5 classifiers (Logistic Regression, KNN, Decision Tree, Naive Bayes, SVC)
- Selected **Logistic Regression** — highest accuracy (86.4%) among the models tested
- Saved the model, scaler, and expected columns with `joblib`
- Deployed as a Streamlit app (`app.py`) for live predictions

**Run the app:**
```bash
cd Projects/Heart-Disease-Prediction
pip install -r ../../requirements.txt
streamlit run app.py
```

---

## 🚗 Car Price Prediction

Predicts a used car's selling price from listing details.

**Location:** `Projects/Car-Price-Prediction/`
**Dataset:** `Car_Price.csv` (4,340 rows × 9 columns)
**Features:** `Brand`, `Model`, `Year`, `KM_Driven`, `Fuel`, `Seller_Type`, `Transmission`, `Owner` → target: `Selling_Price`

**Workflow:**
- Removed duplicates, checked for nulls
- Dropped the high-cardinality `Model` column to avoid one-hot encoding blowup
- One-hot encoded remaining categoricals with `drop_first=True`
- Split data before scaling to prevent data leakage; fit `StandardScaler` on training data only
- Trained a **Linear Regression** model
- Evaluated with R², MAE, RMSE, and a residuals-vs-predicted plot

**Results:** R² = 0.5384, MAE = 180.19k, RMSE = 385.61k

---

## Tech Stack

- Python 3
- pandas, numpy, seaborn, matplotlib
- scikit-learn (`LogisticRegression`, `LinearRegression`, `StandardScaler`, `train_test_split`, metrics)
- streamlit, joblib (deployment)

## How to Run

```bash
git clone https://github.com/asad673-creator/Machine-Learning-Projects.git
cd Machine-Learning-Projects
pip install -r requirements.txt
jupyter notebook   # to explore the notebooks
```

Each project's notebook and (where applicable) Streamlit app live in their own subfolder under `Projects/`.

## Notes

- These are educational/portfolio projects using baseline models — a starting point for further experimentation (regularization, tree-based models, cross-validation, hyperparameter tuning).
- Feel free to fork and extend.

## Disclaimer

The Heart Disease Prediction app is a learning project, not a medical diagnostic tool. Predictions should not be used for real clinical decisions.

## License

Add a license of your choice (e.g. MIT) if you plan to keep this repo public long-term.
