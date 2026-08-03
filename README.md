<div align="center">

# 💎 Diamond Price Prediction

### An end-to-end machine learning system that predicts diamond prices from the 4Cs and physical dimensions

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Lucent%20Frontend-79d7ff?style=for-the-badge)](https://diamond-price-prediction-3-x19d.onrender.com)
[![API Docs](https://img.shields.io/badge/API-FastAPI%20Docs-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://diamond-price-prediction-ukgk.onrender.com/docs)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-EB6C1F?style=flat-square)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com/)

</div>

---

## 📖 Overview

**Diamond Price Prediction** is a full-stack machine learning application that estimates the market value of a diamond from its physical and quality attributes — carat, cut, color, clarity, depth, table, and dimensions (X, Y, Z).

The project covers the complete ML lifecycle:

- 📊 **Exploratory Data Analysis & statistical testing** on a 53,940-row diamond dataset
- 🧹 **Data cleaning** — duplicate removal, IQR-based outlier filtering, skew correction
- 🤖 **Model benchmarking** across 5 regression algorithms with 5-fold cross-validation
- 🎯 **Hyperparameter tuning** via `RandomizedSearchCV` on XGBoost
- 🚀 **Production deployment** as a REST API (FastAPI) with a live, animated frontend

**Live demo:** [diamond-price-prediction-3-x19d.onrender.com](https://diamond-price-prediction-3-x19d.onrender.com)

> ⚠️ The backend is hosted on Render's free tier — the first request after inactivity may take ~30–60 seconds to spin up.

---

## 🏗️ Architecture

```
┌─────────────────────┐         POST /predict          ┌──────────────────────┐
│                      │  ─────────────────────────────▶ │                      │
│   Frontend (HTML/JS) │        JSON: diamond specs       │   FastAPI Backend    │
│   "Lucent" UI        │                                   │   (app.py)           │
│   Hosted on Render   │  ◀───────────────────────────── │   Hosted on Render   │
│                      │        JSON: { "Price": ... }    │                      │
└──────────────────────┘                                   └──────────┬───────────┘
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────────┐
                                                          │  Tuned XGBoost Pipeline  │
                                                          │  (preprocessing + model) │
                                                          │  tuned_xgboost_pipeline  │
                                                          │        .pkl              │
                                                          └──────────────────────────┘
```

---

## ✨ Features

- **Interactive valuation UI** — a dark, jewel-toned interface ("Lucent") with live sliders, an animated 3D-style diamond, and an instant "valuation certificate" result card
- **Real-time inference** — every prediction is a live call to the deployed FastAPI `/predict` endpoint; no values are hard-coded on the frontend
- **Robust preprocessing pipeline** — log-transform + scaling for skewed numeric features, ordinal encoding for `Cut`, one-hot encoding for `Color`/`Clarity`, all bundled into a single `scikit-learn` `Pipeline`/`ColumnTransformer`
- **Tuned gradient-boosted model** — XGBoost regressor optimized with `RandomizedSearchCV` over estimators, learning rate, depth, and sampling ratios
- **Typed, validated API** — request/response schemas enforced with Pydantic, including literal-typed categorical fields (no invalid `Cut`/`Color`/`Clarity` values can reach the model)

---

## 🧠 Model & Data Pipeline

### Dataset
- **Source:** [Diamond Price Prediction Dataset](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset) (Kaggle)
- **Size:** 53,940 rows × 10 columns (carat, cut, color, clarity, depth, table, price, and X/Y/Z dimensions)

### Preprocessing
| Step | Technique |
|---|---|
| Duplicates | Dropped exact duplicate rows |
| Outliers | Removed via IQR method (1.5×IQR bounds) on all numeric columns |
| Skew correction | `log1p` transform + `StandardScaler` on `Carat`, `Depth`, `Y`, `Z` |
| Scaling | `StandardScaler` on `Table`, `X` |
| `Cut` (ordinal) | `OrdinalEncoder` with explicit quality order: `Fair < Good < Very Good < Premium < Ideal` |
| `Color`, `Clarity` (nominal) | `OneHotEncoder` |
| Statistical validation | One-way ANOVA confirmed all three categorical features are significant predictors of price |

### Model selection
Five regressors were benchmarked with 5-fold cross-validated R²:

| Model | CV R² Score |
|---|---|
| Linear Regression | 0.925 |
| Support Vector Machine (SVR) | 0.687 |
| Decision Tree | 0.966 |
| Random Forest | 0.982 |
| **XGBoost** | **0.982** |

XGBoost was selected and further tuned with `RandomizedSearchCV` across `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and `colsample_bytree`.

### Final performance (held-out test set)
| Metric | Score |
|---|---|
| **R² Score** | **0.9838** |
| **Mean Absolute Error** | **$194.16** |

---

## 📂 Repository Structure

```
Diamond-Price-Prediction/
├── Diamond_Price_Prediction.ipynb   # EDA, preprocessing, model training & tuning
├── app.py                           # FastAPI inference service
├── index.html                       # "Lucent" frontend (static, calls the API)
├── requirements.txt                 # Backend dependencies
└── tuned_xgboost_pipeline.pkl       # Serialized sklearn Pipeline (preprocessing + XGBoost)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/akshitgajera1013/Diamond-Price-Prediction.git
cd Diamond-Price-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API locally
```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`, with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### 4. Run the frontend
Simply open `index.html` in a browser, or serve it locally:
```bash
python -m http.server 5500
```
> **Note:** By default `index.html` points at the deployed backend (`https://diamond-price-prediction-ukgk.onrender.com/predict`). To test against your local API, update the `API_URL` constant near the bottom of `index.html`.

### 5. (Optional) Retrain the model
Open `Diamond_Price_Prediction.ipynb` in Jupyter/Colab to reproduce the EDA, preprocessing, and training pipeline from scratch.

---

## 🔌 API Reference

### `POST /predict`

Predicts the price of a diamond given its characteristics.

**Base URL:** `https://diamond-price-prediction-ukgk.onrender.com`

#### Request body

```json
{
  "Carat": 0.9,
  "Cut": "Ideal",
  "Color": "G",
  "Clarity": "VS1",
  "Depth": 61.5,
  "Table": 57.0,
  "X": 6.20,
  "Y": 6.15,
  "Z": 3.80
}
```

| Field | Type | Allowed values |
|---|---|---|
| `Carat` | float | any positive number |
| `Cut` | string | `Ideal`, `Premium`, `Very Good`, `Good`, `Fair` |
| `Color` | string | `D`, `E`, `F`, `G`, `H`, `I`, `J` |
| `Clarity` | string | `IF`, `VVS1`, `VVS2`, `VS1`, `VS2`, `SI1`, `SI2`, `I1` |
| `Depth` | float | depth percentage |
| `Table` | float | table percentage |
| `X` | float | length in mm |
| `Y` | float | width in mm |
| `Z` | float | depth (height) in mm |

#### Response

```json
{
  "Price": 4521.32
}
```

#### Example (cURL)
```bash
curl -X POST "https://diamond-price-prediction-ukgk.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "Carat": 0.9, "Cut": "Ideal", "Color": "G", "Clarity": "VS1",
        "Depth": 61.5, "Table": 57.0, "X": 6.20, "Y": 6.15, "Z": 3.80
      }'
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Modeling** | Python, pandas, NumPy, scikit-learn, XGBoost, SciPy |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | HTML, CSS, vanilla JavaScript |
| **Serialization** | joblib |
| **Deployment** | Render (both frontend and backend) |

---

## 🗺️ Roadmap

- [ ] Add SHAP-based explainability to show which features drove a prediction
- [ ] Add unit/integration tests for the API layer
- [ ] Containerize with Docker for consistent deployment
- [ ] Add a confidence interval / prediction range alongside the point estimate
- [ ] CI/CD pipeline for automated retraining and deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is available under the [MIT License](LICENSE).

---

## 👤 Author

**Akshit Gajera**
GitHub: [@akshitgajera1013](https://github.com/akshitgajera1013)

---

<div align="center">

If you found this project useful, consider giving it a ⭐!

</div>
