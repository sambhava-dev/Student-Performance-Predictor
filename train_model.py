"""
Student Performance Predictor - Model Training Script
College AI/ML Showcase Project

Generates a realistic synthetic student dataset, trains a RandomForestRegressor,
evaluates performance (MAE, RMSE, R²), and saves the trained model bundle to disk.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "student_model.pkl")
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "student_dataset.csv")

FEATURE_NAMES = [
    "study_hours",
    "attendance_percentage",
    "previous_exam_score",
    "assignment_score",
    "screen_time_hours",
    "sleep_hours",
]
TARGET_NAME = "final_exam_score"


def generate_synthetic_data(n_samples: int = 800, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic student records with plausible statistical relationships.
    Variables have natural correlation and realistic noise.
    """
    np.random.seed(random_seed)

    # 1. Study hours per day (range: 0 - 15 hours)
    study_hours = np.random.normal(loc=4.5, scale=2.2, size=n_samples)
    study_hours = np.clip(study_hours, 0.0, 15.0).round(1)

    # 2. Attendance percentage (range: 0 - 100%)
    # Students who study more tend to have slightly higher attendance
    attendance_base = 75.0 + 1.5 * (study_hours - 4.5) + np.random.normal(0, 10, n_samples)
    attendance_percentage = np.clip(attendance_base, 35.0, 100.0).round(1)

    # 3. Previous exam score (range: 0 - 100)
    previous_score_base = 68.0 + 1.8 * (study_hours - 4.5) + 0.1 * (attendance_percentage - 75) + np.random.normal(0, 11, n_samples)
    previous_exam_score = np.clip(previous_score_base, 25.0, 99.0).round(1)

    # 4. Assignment score (range: 0 - 100)
    assignment_base = (
        0.55 * previous_exam_score
        + 0.25 * attendance_percentage
        + 1.2 * study_hours
        + np.random.normal(0, 6, n_samples)
    )
    assignment_score = np.clip(assignment_base, 30.0, 100.0).round(1)

    # 5. Screen time hours per day (range: 0 - 15 hours)
    screen_time = np.random.normal(loc=4.2, scale=1.8, size=n_samples)
    screen_time = np.clip(screen_time, 0.5, 14.5).round(1)

    # 6. Sleep hours per night (range: 0 - 12 hours)
    sleep_hours = np.random.normal(loc=7.0, scale=1.2, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 3.5, 11.5).round(1)

    # 7. Final Exam Score Calculation with realistic educational weights & interactions
    # Prior academic mastery has the strongest weight, followed by continuous effort & attendance.
    # Excessive screen time and severe sleep deprivation create subtle negative drags.
    base_score = (
        0.34 * previous_exam_score
        + 0.26 * assignment_score
        + 1.90 * study_hours
        + 0.16 * attendance_percentage
        - 0.85 * screen_time
        + 1.10 * (sleep_hours - 4.0)
    )

    # Non-linear lifestyle adjustments
    # Penalty for extreme sleep deficit (< 5.0 hours)
    sleep_penalty = np.where(sleep_hours < 5.0, -3.5, 0.0)
    # Penalty for excessive screen time (> 7.5 hours)
    screen_penalty = np.where(screen_time > 7.5, -2.5, 0.0)
    # Synergy bonus for high attendance (> 90%) and solid study hours (> 6.0)
    synergy_bonus = np.where((attendance_percentage > 90.0) & (study_hours > 6.0), 2.5, 0.0)

    # Realistic random exam variance (day-of-exam factors, question variations)
    exam_noise = np.random.normal(loc=0.0, scale=3.2, size=n_samples)

    raw_final_score = base_score + sleep_penalty + screen_penalty + synergy_bonus + exam_noise
    final_exam_score = np.clip(raw_final_score, 10.0, 100.0).round(1)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "previous_exam_score": previous_exam_score,
        "assignment_score": assignment_score,
        "screen_time_hours": screen_time,
        "sleep_hours": sleep_hours,
        "final_exam_score": final_exam_score,
    })

    return df


def train_and_save_model():
    """Train the RandomForestRegressor model, evaluate metrics, and save bundle."""
    print("=" * 60)
    print("🎓 Student Performance Predictor - Model Training")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Generate dataset
    print(f"[*] Generating 800 synthetic student records with realistic relationships...")
    df = generate_synthetic_data(n_samples=800, random_seed=42)
    df.to_csv(DATA_FILE, index=False)
    print(f"[✓] Saved dataset to {DATA_FILE}")

    # 2. Features and target separation
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # 3. 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[*] Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples")

    # 4. Train RandomForestRegressor
    print("[*] Training RandomForestRegressor (100 estimators)...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[✓] Model training completed successfully.")

    # 5. Evaluate on test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "-" * 40)
    print("📊 Evaluation Metrics (Test Set):")
    print(f"   • Mean Absolute Error (MAE):     {mae:.2f}")
    print(f"   • Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   • R² Score (Variance Explained):  {r2:.4f}")
    print("-" * 40)

    # Feature importances
    importances = dict(
        zip(FEATURE_NAMES, [round(float(v), 4) for v in model.feature_importances_])
    )
    print("\n🔍 Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        print(f"   • {feat:<25}: {imp * 100:.2f}%")

    # 6. Bundle model, metadata, metrics, and test sample for dashboard visualization
    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "metrics": {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(len(df)),
        "feature_importances": importances,
        "test_actual": [float(v) for v in y_test.tolist()[:120]],
        "test_predicted": [float(round(v, 1)) for v in y_pred.tolist()[:120]],
    }

    joblib.dump(bundle, MODEL_FILE)
    print(f"\n[✓] Saved complete model bundle to {MODEL_FILE}")
    print("=" * 60)
    print("Ready to run Streamlit app:")
    print("   streamlit run app.py")
    print("=" * 60)

    return bundle


if __name__ == "__main__":
    train_and_save_model()
