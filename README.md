# Student Performance Predictor

> **AI-powered academic performance estimation**  
> An educational Machine Learning showcase application designed for college demonstrations, coursework presentations, and viva discussions.

---

## 📌 Overview

**Student Performance Predictor** is a complete, working Python and Machine Learning project that demonstrates how regression algorithms can estimate a student's expected final examination score (on a 0–100 scale) using six measurable academic and lifestyle indicators.

The project emphasizes clean software engineering, statistical transparency, and uncertainty quantification (reporting Mean Absolute Error bounds alongside point predictions) while avoiding over-complicated dependencies or opaque deep-learning models.

---

## ✨ Features

- **Interactive Academic Dashboard:** Clean, professional white/light minimalist interface built with Streamlit.
- **6-Factor Input Controls:**
  - **Study Hours:** Daily dedicated self-study duration (0.0 – 15.0 hours, step 0.5)
  - **Attendance Percentage:** Lecture and lab participation (0 – 100%, step 1%)
  - **Previous Exam Score:** Prior midterm examination marks (0 – 100, step 1)
  - **Assignment Score:** Continuous coursework evaluation (0 – 100, step 1)
  - **Screen Time:** Daily leisure screen/device usage (0.0 – 15.0 hours, step 0.5)
  - **Sleep Hours:** Nightly average rest duration (0.0 – 12.0 hours, step 0.5)
- **Instant ML Inference:**
  - High-precision point score calculation (e.g., `76.4 / 100`)
  - Expected score range accounting for model MAE (e.g., `72.8 – 80.0 / 100`)
  - 5-tier academic classification:
    - `90 – 100` : **Excellent**
    - `75 – 89` : **Good**
    - `60 – 74` : **Average**
    - `40 – 59` : **Needs Improvement**
    - `0 – 39` : **At Risk**
- **Normalized Factor Visualization:** Matplotlib horizontal bar chart comparing student inputs on a standardized relative percentage scale.
- **Student Profile Synthesis:** Plain-English summary of academic preparation, attendance consistency, and lifestyle indicators.
- **Model Transparency & Diagnostics:**
  - Live display of training samples (80%) vs testing samples (20%)
  - Regression metrics: MAE, RMSE, and $R^2$ Score
  - Scatter plot of Actual vs Predicted test scores with $y = x$ reference line
  - Gini feature importance breakdown
- **Robust Error Handling & Caching:** Graceful handling of uninitialized models, one-click in-app retraining, input boundaries validation, and `@st.cache_resource` loading.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+ / 3.11+** | Core runtime environment |
| **Pandas** | Tabular data representation and dataset serialization |
| **NumPy** | Statistical distributions, noise modeling, and matrix computations |
| **Scikit-learn** | Model training (`RandomForestRegressor`), evaluation metrics (`MAE`, `RMSE`, `R2`) |
| **Streamlit** | Reactive web application framework and user interface |
| **Matplotlib** | Academic visualizations and diagnostic charts |
| **Joblib** | Serialization and loading of model weights and metadata |

---

## 🧠 Machine Learning Approach

### 1. Synthetic Dataset Generation (`train_model.py`)
To ensure reproducibility without depending on third-party API keys or external servers, `train_model.py` generates a synthetic dataset of **800 student records** featuring realistic statistical properties:
- Prior examination scores and assignment scores provide primary predictive weight.
- Study hours and high attendance exhibit positive non-linear synergies.
- Sleep deprivation (< 5 hours) and excessive screen time (> 7.5 hours) introduce subtle penalties.
- Normal Gaussian noise simulates real-world examination day variance.

### 2. Model Selection: Random Forest Regressor
A **Random Forest Regressor** (100 estimators, max depth 10, random state 42) was selected because:
- **Ensemble Stability:** Combines predictions from multiple decorrelated decision trees (bagging) to minimize overfitting.
- **Non-Linear Relationships:** Naturally captures complex interactions between sleep, attendance, and study hours without manual polynomial feature engineering.
- **Interpretability:** Provides intrinsic feature importances (Gini impurity reduction) easily explained during a college viva.

### 3. Evaluation Metrics
Evaluated on a 20% holdout test set (160 unseen student records):
- **Mean Absolute Error (MAE):** Quantifies average point deviation in exam marks.
- **Root Mean Squared Error (RMSE):** Measures variance with heavier penalty on severe outliers.
- **$R^2$ Score (Coefficient of Determination):** Explains the proportion of variance in exam scores captured by the model.

---

## 📁 Project Structure

```text
student-performance-predictor/
├── app.py                  # Streamlit web application & user interface
├── train_model.py          # Dataset generator, training pipeline & evaluator
├── requirements.txt        # Minimal Python package requirements
├── README.md               # Project documentation and presentation guide
├── data/
│   └── student_dataset.csv # Generated 800-record student dataset (auto-created)
└── model/
    └── student_model.pkl   # Serialized model bundle & metadata (auto-created)
```

---

## 🚀 Installation & Running Instructions

### Step 1: Clone or Navigate to the Directory
```bash
cd student-performance-predictor
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model
```bash
python train_model.py
```
*This generates `data/student_dataset.csv`, trains the Random Forest model, logs metrics to console, and produces `model/student_model.pkl`.*

### Step 5: Launch the Streamlit Application
```bash
streamlit run app.py
```
*Your browser will automatically open to `http://localhost:8501` (or local port).*

---

## 💡 Example Prediction

**Input Profile:**
- Study Hours: `5.5 hrs/day`
- Attendance: `90%`
- Previous Exam Score: `78 / 100`
- Assignment Score: `82 / 100`
- Screen Time: `3.0 hrs/day`
- Sleep Hours: `7.5 hrs/night`

**Prediction Output:**
- **Predicted Score:** `82.6 / 100`
- **Performance Category:** `Good`
- **Expected Score Range:** `79.2 – 86.0 / 100` (within MAE confidence bound)
- **Student Profile:** Strong baseline performance and high attendance consistency.

---

## 🔮 Future Improvements

1. **Feature Expansion:** Incorporate qualitative features such as participation in extracurricular activities, commute duration, and study group involvement.
2. **Hyperparameter Tuning:** Demonstrate `GridSearchCV` or `RandomizedSearchCV` cross-validation during viva.
3. **SHAP / LIME Explanations:** Integrate local interpretability plots to explain individual prediction weight contributions per student.
4. **Time-Series Tracking:** Track score progression across consecutive quarters rather than a single terminal exam.

---

## ⚖️ Disclaimer

This project is intended strictly for educational and demonstration purposes. Predictions are statistical estimates generated by a machine-learning model trained on synthetic data and should not be treated as definitive or authoritative assessments of a student's actual cognitive ability, future career outcomes, or individual potential.
