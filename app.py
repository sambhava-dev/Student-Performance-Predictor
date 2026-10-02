"""
Student Performance Predictor
College AI/ML Showcase Project
Built with Python, Scikit-learn, Streamlit, and Matplotlib.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page settings with an academic, clean layout
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join("model", "student_model.pkl")

# Minimal, clean academic CSS styling
st.markdown(
    """
    <style>
    /* Clean white/light academic styling */
    .main {
        background-color: #fcfcfd;
    }
    .stApp {
        color: #1f2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    /* Minimal Card container */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .result-score-banner {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2563eb;
        border-radius: 8px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
    }
    .badge-excellent {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-good {
        display: inline-block;
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-average {
        display: inline-block;
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-needs-improvement {
        display: inline-block;
        background-color: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-at-risk {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained ML model...")
def load_trained_model():
    """Load the trained model bundle with caching."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as exc:
        st.error(f"Error loading model from {MODEL_PATH}: {exc}")
        return None


def classify_score(score: float):
    """
    Classify score based on requirements:
    90–100 -> Excellent
    75–89  -> Good
    60–74  -> Average
    40–59  -> Needs Improvement
    0–39   -> At Risk
    """
    if score >= 90.0:
        return "Excellent", "badge-excellent", "Consistent high achievement across academic preparation and diligence."
    elif score >= 75.0:
        return "Good", "badge-good", "Strong grasp of syllabus material with solid coursework participation."
    elif score >= 60.0:
        return "Average", "badge-average", "Meets basic subject criteria; potential for growth with targeted study hours."
    elif score >= 40.0:
        return "Needs Improvement", "badge-needs-improvement", "At-risk indicators present in attendance or core assessments."
    else:
        return "At Risk", "badge-at-risk", "Immediate academic intervention recommended to reverse compounding deficits."


def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.markdown("### 🎓 Student Performance Predictor")
        st.caption("AI-powered academic performance estimation")
        st.markdown("---")

        nav_choice = st.radio(
            "Navigation",
            options=["Prediction", "Model Information", "About Project"],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #6b7280; line-height: 1.4;">
                <strong>Project Scope:</strong><br>
                Undergraduate AI/ML Demonstration<br>
                Algorithm: Random Forest Regressor<br>
                Framework: Streamlit & Scikit-learn
            </div>
            """,
            unsafe_allow_html=True,
        )

        return nav_choice


def render_prediction_page(bundle):
    """Render the main prediction interface."""
    st.markdown("## Student Performance Predictor")
    st.markdown(
        "<p style='color: #4b5563; font-size: 1.05rem; margin-top: -10px;'>"
        "Estimate a student's expected final examination score using machine learning."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if bundle is None:
        st.warning(
            "⚠️ **Trained Model Not Found**\n\n"
            f"The model bundle (`{MODEL_PATH}`) could not be located. "
            "Please train the model first by running:\n\n"
            "```bash\npython train_model.py\n```"
        )
        if st.button("🚀 Train Model Now In-App", type="primary"):
            with st.spinner("Training Random Forest Regressor..."):
                try:
                    import train_model
                    train_model.train_and_save_model()
                    st.success("✓ Model trained and saved successfully! Refreshing...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
        return

    # Two column layout for input controls
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📚 Academic Factors")
        study_hours = st.slider(
            "Study Hours (Daily)",
            min_value=0.0,
            max_value=15.0,
            value=4.5,
            step=0.5,
            help="Average dedicated self-study hours outside class (range: 0 - 15h).",
        )

        attendance = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=85,
            step=1,
            help="Semester lecture and tutorial attendance percentage (0 - 100%).",
        )

        previous_score = st.slider(
            "Previous Exam Score",
            min_value=0,
            max_value=100,
            value=72,
            step=1,
            help="Student's marks in preceding midterm/quarterly examination (0 - 100).",
        )

    with col2:
        st.markdown("#### 🌿 Coursework & Lifestyle Factors")
        assignment_score = st.slider(
            "Assignment Score",
            min_value=0,
            max_value=100,
            value=75,
            step=1,
            help="Cumulative grade from assignments, projects, and lab submissions (0 - 100).",
        )

        screen_time = st.slider(
            "Screen Time (Daily Non-Academic Hours)",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.5,
            help="Leisure screen time spent on gaming, streaming, or social media (0 - 15h).",
        )

        sleep_hours = st.slider(
            "Sleep Hours (Nightly Average)",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            help="Average nightly sleep duration during semester weeks (0 - 12h).",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Performance", type="primary", use_container_width=True)

    # Initialize or preserve state
    if "predicted" not in st.session_state:
        st.session_state["predicted"] = False

    if predict_clicked:
        st.session_state["predicted"] = True
        st.session_state["inputs"] = {
            "study_hours": float(study_hours),
            "attendance_percentage": float(attendance),
            "previous_exam_score": float(previous_score),
            "assignment_score": float(assignment_score),
            "screen_time_hours": float(screen_time),
            "sleep_hours": float(sleep_hours),
        }

    if st.session_state.get("predicted", False):
        inputs = st.session_state["inputs"]
        input_df = pd.DataFrame([inputs])

        # Predict using Random Forest model
        model = bundle["model"]
        predicted_score = float(model.predict(input_df)[0])
        predicted_score = max(0.0, min(100.0, predicted_score))
        mae = bundle["metrics"]["mae"]

        lower_bound = max(0.0, round(predicted_score - mae, 1))
        upper_bound = min(100.0, round(predicted_score + mae, 1))

        category, badge_class, category_desc = classify_score(predicted_score)

        st.markdown("<hr style='margin: 28px 0 20px 0;'>", unsafe_allow_html=True)

        # 6. Large Result Card
        st.markdown(
            f"""
            <div class="result-score-banner">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">
                    Predicted Final Score
                </div>
                <div style="display: flex; align-items: baseline; gap: 16px; margin: 8px 0 12px 0;">
                    <span style="font-size: 3.2rem; font-weight: 700; color: #0f172a; line-height: 1;">
                        {predicted_score:.1f}
                    </span>
                    <span style="font-size: 1.4rem; color: #94a3b8; font-weight: 500;">
                        / 100
                    </span>
                    <span class="{badge_class}" style="margin-left: 12px;">
                        {category}
                    </span>
                </div>
                <div style="font-size: 0.95rem; color: #334155;">
                    <strong>Expected Score Range:</strong> {lower_bound} – {upper_bound} / 100 
                    <span style="color: #64748b; font-size: 0.85rem;">(± {mae:.1f} MAE uncertainty margin)</span>
                </div>
                <div style="font-size: 0.88rem; color: #64748b; margin-top: 6px;">
                    {category_desc}
                </div>
                <div class="disclaimer-box">
                    <strong>Educational ML Demonstration:</strong> This prediction is a statistical estimate produced by a machine-learning regression model based on synthetic training data. It is intended purely for instructional demonstration and does not constitute a medically or scientifically certain evaluation of individual student capability.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 7. Visualization & Student Profile
        st.markdown("### 📊 Student Input Factor Analysis")
        col_chart, col_profile = st.columns([3, 2], gap="large")

        with col_chart:
            st.markdown("##### Normalized Factor Comparison")
            # Normalize factors on a 0-100% scale for fair visual comparison
            # Study hours (max 15), Attendance (100), Prev Exam (100), Assignment (100), Screen Time (max 15), Sleep (max 12)
            factor_labels = [
                f"Study Hours ({inputs['study_hours']}h / 15h)",
                f"Attendance ({int(inputs['attendance_percentage'])}%)",
                f"Prev Exam ({int(inputs['previous_exam_score'])}/100)",
                f"Assignment ({int(inputs['assignment_score'])}/100)",
                f"Screen Time ({inputs['screen_time_hours']}h / 15h)",
                f"Sleep ({inputs['sleep_hours']}h / 12h)",
            ]
            normalized_vals = [
                min(100.0, (inputs["study_hours"] / 15.0) * 100),
                inputs["attendance_percentage"],
                inputs["previous_exam_score"],
                inputs["assignment_score"],
                min(100.0, (inputs["screen_time_hours"] / 15.0) * 100),
                min(100.0, (inputs["sleep_hours"] / 12.0) * 100),
            ]

            # Colors: neutral blues and slate
            colors = [
                "#3b82f6" if val >= 50 else "#93c5fd"
                for val in normalized_vals
            ]
            # Emphasize screen time with an amber tint if excessive
            if inputs["screen_time_hours"] > 7.0:
                colors[4] = "#f59e0b"

            fig, ax = plt.subplots(figsize=(7, 3.8))
            y_pos = np.arange(len(factor_labels))
            bars = ax.barh(y_pos, normalized_vals, color=colors, height=0.55, edgecolor="#e2e8f0", linewidth=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(factor_labels, fontsize=9.5, color="#1e293b")
            ax.invert_yaxis()  # top-down reading order
            ax.set_xlim(0, 105)
            ax.set_xlabel("Relative Scale (%)", fontsize=9, color="#64748b")
            ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e2e8f0")
            ax.spines["bottom"].set_color("#e2e8f0")

            # Value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color="#475569",
                    fontweight="500",
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_profile:
            st.markdown("##### Student Profile")

            # Academic preparation
            acad_avg = (inputs["previous_exam_score"] + inputs["assignment_score"]) / 2.0
            if acad_avg >= 80:
                acad_text = "Strong baseline performance indicated by prior exam and assignment scores."
            elif acad_avg >= 60:
                acad_text = "Moderate baseline performance; foundational topics are partially consolidated."
            else:
                acad_text = "Prior assessment indicators suggest recurring conceptual gaps that require review."

            # Attendance
            att_val = inputs["attendance_percentage"]
            if att_val >= 85:
                att_text = "High lecture and lab attendance, supporting continuous curriculum continuity."
            elif att_val >= 70:
                att_text = "Satisfactory attendance with minor gaps in classroom engagement."
            else:
                att_text = "Sub-optimal attendance below 70%, which historically correlates with missed topics."

            # Lifestyle balance
            slp = inputs["sleep_hours"]
            scr = inputs["screen_time_hours"]
            if slp >= 6.5 and scr <= 5.0:
                life_text = "Balanced rest and leisure ratio, favorable for cognitive retention and focus."
            elif slp < 6.0 and scr > 6.0:
                life_text = "Elevated leisure screen time coupled with reduced rest hours observed."
            elif slp < 6.0:
                life_text = "Rest hours are below recommended levels for sustained exam revision."
            else:
                life_text = "Standard lifestyle indicators without pronounced sleep or screen time skew."

            st.markdown(
                f"""
                <div class="metric-card">
                    <strong style="color: #1e3a8a; font-size: 0.95rem;">📖 Academic Preparation</strong>
                    <p style="font-size: 0.88rem; color: #475569; margin: 4px 0 12px 0;">{acad_text}</p>
                    
                    <strong style="color: #1e3a8a; font-size: 0.95rem;">📅 Attendance Consistency</strong>
                    <p style="font-size: 0.88rem; color: #475569; margin: 4px 0 12px 0;">{att_text}</p>
                    
                    <strong style="color: #1e3a8a; font-size: 0.95rem;">⚖️ Lifestyle Indicators</strong>
                    <p style="font-size: 0.88rem; color: #475569; margin: 4px 0 4px 0;">{life_text}</p>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-style: italic;">
                    * Note: Profile insights are descriptive summaries of inputs and do not imply direct clinical or psychological causation.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_info_page(bundle):
    """Render the Model Information page."""
    st.markdown("## Model Information")
    st.markdown(
        "<p style='color: #4b5563; font-size: 1.05rem; margin-top: -10px;'>"
        "Architecture, statistical validation metrics, and training specifications."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if bundle is None:
        st.warning("Model bundle not loaded. Please run `python train_model.py` first.")
        return

    metrics = bundle["metrics"]
    train_n = bundle.get("train_samples", 640)
    test_n = bundle.get("test_samples", 160)
    total_n = bundle.get("total_samples", train_n + test_n)

    # Core statistics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 600;">MODEL TYPE</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-top: 4px;">Random Forest</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">100 Decision Trees</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 600;">MEAN ABS. ERROR</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #2563eb; margin-top: 4px;">{metrics['mae']:.2f} pts</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Average deviation from actual</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 600;">ROOT MEAN SQ. ERROR</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #2563eb; margin-top: 4px;">{metrics['rmse']:.2f} pts</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Penalizes larger outliers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 600;">R² SCORE</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #059669; margin-top: 4px;">{metrics['r2']:.4f}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">{(metrics['r2'] * 100):.1f}% variance explained</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset details
    st.markdown("#### 📂 Dataset & Split Specifications")
    st.markdown(
        f"""
        - **Dataset:** Synthetic student dataset ({total_n} total generated instances)
        - **Training samples:** **{train_n}** records (80% split)
        - **Testing samples:** **{test_n}** records (20% holdout split)
        - **Input Features (6):** `study_hours`, `attendance_percentage`, `previous_exam_score`, `assignment_score`, `screen_time_hours`, `sleep_hours`
        - **Target Variable (1):** `final_exam_score` (Continuous scale: 0 – 100)
        """
    )

    st.markdown("---")

    # Visualizations: Actual vs Predicted & Feature Importances
    col_vis1, col_vis2 = st.columns(2, gap="large")

    with col_vis1:
        st.markdown("#### Actual vs Predicted Test Scores")
        actuals = bundle.get("test_actual", [])
        preds = bundle.get("test_predicted", [])

        if actuals and preds:
            fig, ax = plt.subplots(figsize=(6, 4.8))
            ax.scatter(actuals, preds, alpha=0.65, color="#2563eb", edgecolors="#1d4ed8", s=36, label="Test Instances")

            # Reference ideal fit line (y = x)
            min_val = min(min(actuals), min(preds)) - 3
            max_val = max(max(actuals), max(preds)) + 3
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#dc2626", linewidth=1.5, label="Ideal Fit (y = x)")

            ax.set_xlabel("Actual Exam Score", fontsize=9.5, color="#334155")
            ax.set_ylabel("Predicted Exam Score", fontsize=9.5, color="#334155")
            ax.set_title("Test Set Evaluation (80/20 Holdout)", fontsize=10.5, pad=10, color="#0f172a", fontweight="600")
            ax.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cbd5e1")
            ax.spines["bottom"].set_color("#cbd5e1")
            ax.legend(frameon=True, fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Test set predictions not available in bundle.")

    with col_vis2:
        st.markdown("#### Random Forest Feature Importance")
        importances = bundle.get("feature_importances", {})
        if importances:
            sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            feat_names = [f[0].replace("_", " ").title() for f in sorted_feats]
            feat_weights = [f[1] * 100 for f in sorted_feats]

            fig, ax = plt.subplots(figsize=(6, 4.8))
            y_pos = np.arange(len(feat_names))
            bars = ax.barh(y_pos, feat_weights, color="#0284c7", height=0.55, edgecolor="#e2e8f0")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_names, fontsize=9.5, color="#1e293b")
            ax.invert_yaxis()
            ax.set_xlabel("Relative Importance (%)", fontsize=9.5, color="#334155")
            ax.set_title("Gini Impurity Reduction Contribution", fontsize=10.5, pad=10, color="#0f172a", fontweight="600")
            ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cbd5e1")
            ax.spines["bottom"].set_color("#cbd5e1")

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, color="#475569")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    # Explanatory card
    st.markdown(
        """
        <div class="metric-card">
            <h5 style="color: #0f172a; margin-top: 0;">🌲 How Random Forest Works (Viva Concept)</h5>
            <p style="font-size: 0.95rem; color: #334155; line-height: 1.6; margin-bottom: 8px;">
                <em>"Random Forest combines predictions from many decision trees to produce a more stable prediction."</em>
            </p>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                A single decision tree is prone to high variance and overfitting on idiosyncratic samples. 
                Random Forest employs <strong>bootstrap aggregating (bagging)</strong>: it creates 100 individual decision trees, 
                each trained on a random subsample of student records with random feature subsets at every split. 
                The regression prediction is the arithmetic average across all trees, significantly reducing prediction variance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page():
    """Render the About Project page."""
    st.markdown("## About the Project")
    st.markdown(
        "<p style='color: #4b5563; font-size: 1.05rem; margin-top: -10px;'>"
        "Student Performance Predictor is an educational AI/ML application that demonstrates how machine learning can be used to estimate academic performance from selected student-related features."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🛠️ Technologies")
        st.markdown(
            """
            * **Python:** Core programming language (version 3.10+)
            * **Pandas:** Tabular data representation and feature preprocessing
            * **NumPy:** Numerical matrix operations and statistical noise simulation
            * **Scikit-learn:** Train/test split, `RandomForestRegressor`, and evaluation metrics
            * **Streamlit:** Interactive web application and reactive dashboard UI
            * **Matplotlib:** Static chart generation for factor visualization and model validation
            * **Joblib:** Fast serialization of trained models and metadata bundles
            """
        )

        st.markdown("### ⚖️ Disclaimer")
        st.markdown(
            """
            <div class="disclaimer-box" style="font-size: 0.9rem; line-height: 1.5;">
                "This project is intended for educational and demonstration purposes. Predictions are estimates generated by a machine-learning model and should not be treated as definitive assessments of a student's actual ability or future performance."
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 🎯 Educational Objectives")
        st.markdown(
            """
            This showcase was created for college viva demonstrations and classroom ML case studies:
            
            1. **Supervised Regression:** Mapping 6 continuous & ordinal features to a continuous output range [0, 100].
            2. **Ensemble Modeling:** Demonstrating why bagging outperforms single-tree estimators in generalization.
            3. **Uncertainty Communication:** Pairing point predictions with Mean Absolute Error confidence intervals rather than presenting single uncalibrated values.
            4. **Clean Code Structure:** Maintaining modular separation between dataset generation/training (`train_model.py`) and UI presentation (`app.py`).
            """
        )

        st.markdown("### 🧪 Quick Verification Commands")
        st.code(
            """# 1. Install required packages
pip install -r requirements.txt

# 2. Train and validate model
python train_model.py

# 3. Launch interactive dashboard
streamlit run app.py
""",
            language="bash",
        )


def main():
    """Main application entry point."""
    bundle = load_trained_model()
    selected_page = render_sidebar()

    if selected_page == "Prediction":
        render_prediction_page(bundle)
    elif selected_page == "Model Information":
        render_model_info_page(bundle)
    elif selected_page == "About Project":
        render_about_page()


if __name__ == "__main__":
    main()
