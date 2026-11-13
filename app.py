"""
Interactive Streamlit Web Application.
Multi-Task Prediction for Student GPA, Dropout Risk, and Academic Performance Level.
Production-ready for Streamlit Community Cloud and Local Execution.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

# ==========================================================
# Page Configuration & Clean English Typography
# ==========================================================
st.set_page_config(
    page_title="Student Academic GPA & Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Card styling */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Advisory / Recommendation Boxes */
    .recommendation-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
        color: #166534;
    }
    
    .danger-box {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
        color: #991b1b;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


@st.cache_resource(show_spinner="Loading deep learning model and pre-trained artifacts...")
def load_all_artifacts():
    """Load pre-trained model and metadata with cached resource."""
    model_path = os.path.join(ARTIFACTS_DIR, "model.keras")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    y_scaler_path = os.path.join(ARTIFACTS_DIR, "y_scaler.pkl")
    cols_path = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
    stats_path = os.path.join(ARTIFACTS_DIR, "feature_stats.json")
    metrics_path = os.path.join(ARTIFACTS_DIR, "test_metrics.json")

    if not all(
        os.path.exists(p)
        for p in [model_path, scaler_path, y_scaler_path, cols_path, stats_path, metrics_path]
    ):
        return None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    with open(cols_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(stats_path, "r", encoding="utf-8") as f:
        feature_stats = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    return model, scaler, y_scaler, feature_columns, feature_stats, test_metrics


artifacts = load_all_artifacts()

if artifacts is None:
    st.error(
        "⚠️ Model artifacts not found.\n\n"
        "To generate production artifacts, please run the following command in your terminal:\n\n"
        "`python -m src.train`"
    )
    st.stop()

model, scaler, y_scaler, feature_columns, feature_stats, test_metrics = artifacts


# ==========================================================
# Helper Function for Generating Form Controls
# ==========================================================
def make_slider(col_name: str, label: str, section_prefix: str, step: float = None):
    """Generate safe numeric sliders based on dataset statistics."""
    stats = feature_stats[col_name]
    is_int = stats["dtype"] == "int"
    min_v = stats["min"]
    max_v = stats["max"]
    default_v = stats["mean"]

    # Use session state preset value if available
    session_key = f"{section_prefix}_{col_name}"
    if session_key in st.session_state:
        default_v = st.session_state[session_key]

    if is_int:
        min_v = int(min_v)
        max_v = int(max_v)
        default_v = int(round(default_v))
        step = int(step or 1)
    else:
        min_v = float(round(min_v, 2))
        max_v = float(round(max_v, 2))
        default_v = float(round(default_v, 2))
        step = float(step or 0.1)

    # Ensure default is within bounds
    default_v = max(min_v, min(max_v, default_v))

    return st.slider(
        label=label,
        min_value=min_v,
        max_value=max_v,
        value=default_v,
        step=step,
        key=session_key,
    )


# ==========================================================
# Sidebar: Presets & Model Information
# ==========================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=600&q=80",
        caption="Student Success & Academic Analytics Platform",
        use_container_width=True,
    )
    st.markdown("### 🎯 Quick Test Profiles")
    st.caption("Click a preset profile to automatically populate the input form with simulated student data:")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("🌟 High Achiever", use_container_width=True):
            st.session_state["p1_CurrentGPA"] = 18.5
            st.session_state["p2_HighSchoolGPA"] = 19.0
            st.session_state["p2_StudyHoursPerWeek"] = 30
            st.session_state["p2_AttendanceRate"] = 95.0
            st.session_state["p2_PreviousFailures"] = 0
            st.session_state["p3_LoginFrequency"] = 28
            st.session_state["p3_CompletionRate"] = 92.0
            st.session_state["p3_AveragePracticeScore"] = 90.0
            st.session_state["p3_ConsecutiveInactiveDays"] = 1
            st.rerun()

    with p_col2:
        if st.button("⚠️ At-Risk Student", use_container_width=True):
            st.session_state["p1_CurrentGPA"] = 10.5
            st.session_state["p2_HighSchoolGPA"] = 11.0
            st.session_state["p2_StudyHoursPerWeek"] = 7
            st.session_state["p2_AttendanceRate"] = 55.0
            st.session_state["p2_PreviousFailures"] = 3
            st.session_state["p3_LoginFrequency"] = 4
            st.session_state["p3_CompletionRate"] = 35.0
            st.session_state["p3_AveragePracticeScore"] = 45.0
            st.session_state["p3_ConsecutiveInactiveDays"] = 9
            st.rerun()

    if st.button("🔄 Reset to Defaults", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith(("p1_", "p2_", "p3_")):
                del st.session_state[k]
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ About the Project")
    st.info(
        "This platform employs a **Multi-Task Deep Neural Network (MT-DNN)** to simultaneously forecast "
        "three essential educational targets:\n\n"
        "1. **Future Term GPA** (Regression)\n"
        "2. **Dropout Risk Probability** (Binary Classification)\n"
        "3. **Overall Academic Performance Tier** (4-Class Classification)\n\n"
        "[GitHub Repository](https://github.com/Amin-Ezz/student-success-predictor)"
    )

# ==========================================================
# Main Header & Model Performance Metrics
# ==========================================================
st.title("🎓 Student Academic GPA & Dropout Risk Predictor")
st.markdown(
    "Enter student demographic information, prior academic history, and virtual LMS activity "
    "to forecast upcoming academic outcomes and evaluate potential dropout risks using deep learning."
)

with st.expander("📈 View Model Benchmark Performance on Holdout Test Set (1,000 samples)", expanded=False):
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(
            label="Future GPA R² Score",
            value=f"{test_metrics['future_gpa']['r2']:.1%}",
            delta=f"MAE: {test_metrics['future_gpa']['mae']:.2f}",
        )
    with m_col2:
        st.metric(
            label="Dropout Prediction Accuracy",
            value=f"{test_metrics['dropout']['accuracy']:.1%}",
            delta=f"F1: {test_metrics['dropout']['f1']:.1%}",
        )
    with m_col3:
        st.metric(
            label="Academic Tier Accuracy",
            value=f"{test_metrics['performance']['accuracy']:.1%}",
            delta=f"F1: {test_metrics['performance']['f1']:.1%}",
        )

st.divider()

# ==========================================================
# Input Forms inside Tabs
# ==========================================================
tab_personal, tab_academic, tab_online = st.tabs([
    "👤 Personal & Demographics",
    "📚 Academic Records & History",
    "💻 Online LMS Engagement",
])

user_inputs = {}

with tab_personal:
    st.subheader("Demographic & Socioeconomic Background")
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["Age"] = make_slider("Age", "Student Age", "p1")
        user_inputs["Gender"] = st.selectbox(
            "Gender",
            [0, 1],
            format_func=lambda x: "Male" if x == 1 else "Female",
            key="p1_gender",
        )
        user_inputs["Department"] = st.selectbox(
            "Academic Department / Major",
            [0, 1, 2, 3, 4],
            format_func=lambda x: f"Department {x + 1} (Engineering & Science)",
            key="p1_dept",
        )
        user_inputs["Semester"] = make_slider("Semester", "Current Semester", "p1")

    with c2:
        user_inputs["FamilyIncome"] = make_slider(
            "FamilyIncome", "Estimated Monthly Family Income Index", "p1"
        )
        user_inputs["ParentEducation"] = st.selectbox(
            "Parent Highest Education Level",
            [0, 1, 2, 3],
            format_func=lambda x: [
                "High School or Less",
                "Associate Degree",
                "Bachelor's Degree",
                "Master's or Ph.D.",
            ][x],
            index=2,
            key="p1_pedu",
        )
        user_inputs["Scholarship"] = st.selectbox(
            "Academic Scholarship Status",
            [0, 1],
            format_func=lambda x: "No Scholarship" if x == 0 else "Scholarship Recipient",
            key="p1_schol",
        )

with tab_academic:
    st.subheader("Prior Academic Performance & Study Habits")
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["HighSchoolGPA"] = make_slider("HighSchoolGPA", "High School GPA (Scale 0-20)", "p2")
        user_inputs["CurrentGPA"] = make_slider("CurrentGPA", "Current Cumulative GPA (Scale 0-20)", "p2")
        user_inputs["StudyHoursPerWeek"] = make_slider("StudyHoursPerWeek", "Weekly Study Hours", "p2")
        user_inputs["SleepHours"] = make_slider("SleepHours", "Average Daily Sleep Hours", "p2")
        user_inputs["AttendanceRate"] = make_slider("AttendanceRate", "Class Attendance Rate (%)", "p2")
        user_inputs["LibraryVisits"] = make_slider("LibraryVisits", "Monthly Library Visits", "p2")

    with c2:
        user_inputs["AssignmentAverage"] = make_slider("AssignmentAverage", "Assignment Average Score (%)", "p2")
        user_inputs["QuizAverage"] = make_slider("QuizAverage", "Quiz Average Score (%)", "p2")
        user_inputs["MidtermScore"] = make_slider("MidtermScore", "Midterm Exam Score (%)", "p2")
        user_inputs["ClassParticipation"] = make_slider("ClassParticipation", "Class Participation Score (%)", "p2")
        user_inputs["PreviousFailures"] = make_slider("PreviousFailures", "Number of Previous Failed Courses", "p2")
        user_inputs["InternetUsage"] = make_slider("InternetUsage", "Daily Non-Academic Internet Usage (Hours)", "p2")

with tab_online:
    st.subheader("LMS Platform Engagement & Learning Analytics")
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["LoginFrequency"] = make_slider("LoginFrequency", "Weekly LMS Login Frequency", "p3")
        user_inputs["TotalLoginTime"] = make_slider("TotalLoginTime", "Total LMS Time (Minutes/Week)", "p3")
        user_inputs["VideosWatched"] = make_slider("VideosWatched", "Lecture Videos Watched", "p3")
        user_inputs["CompletionRate"] = make_slider("CompletionRate", "Course Material Completion Rate (%)", "p3")
        user_inputs["AveragePracticeScore"] = make_slider("AveragePracticeScore", "Online Practice Exam Average (%)", "p3")
        user_inputs["PracticeTestsTaken"] = make_slider("PracticeTestsTaken", "Number of Practice Tests Taken", "p3")
        user_inputs["DownloadedResources"] = make_slider("DownloadedResources", "Course Materials & Handouts Downloaded", "p3")
        user_inputs["AssignmentSubmissionDelay"] = make_slider("AssignmentSubmissionDelay", "Average Assignment Submission Delay (Days)", "p3")
        user_inputs["ConsecutiveInactiveDays"] = make_slider("ConsecutiveInactiveDays", "Consecutive Inactive Days on LMS", "p3")

    with c2:
        user_inputs["ForumPosts"] = make_slider("ForumPosts", "Discussion Forum Posts Created", "p3")
        user_inputs["QuestionsAsked"] = make_slider("QuestionsAsked", "Forum Questions Asked", "p3")
        user_inputs["QuestionsAnswered"] = make_slider("QuestionsAnswered", "Peer Questions Answered", "p3")
        user_inputs["MobileUsage"] = make_slider("MobileUsage", "Mobile Device Access Ratio (%)", "p3")
        user_inputs["DesktopUsage"] = make_slider("DesktopUsage", "Desktop / Laptop Access Ratio (%)", "p3")
        user_inputs["NightStudyPercentage"] = make_slider("NightStudyPercentage", "Night Study Ratio (12 AM - 6 AM) (%)", "p3")
        user_inputs["WeekendStudyPercentage"] = make_slider("WeekendStudyPercentage", "Weekend Study Ratio (%)", "p3")
        user_inputs["AIUsageFrequency"] = make_slider("AIUsageFrequency", "AI Learning Tools Usage Frequency", "p3")
        user_inputs["CourseCompletionTime"] = make_slider("CourseCompletionTime", "Estimated Course Completion Pace (Weeks)", "p3")
        user_inputs["CertificateEarned"] = st.selectbox(
            "Course Certificate Earned",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="p3_cert",
        )

# Ensure StudentID default is preserved
user_inputs["StudentID"] = feature_stats["StudentID"]["mean"]

st.divider()

# ==========================================================
# Run Prediction
# ==========================================================
col_btn, _ = st.columns([1, 2])
with col_btn:
    run_prediction = st.button("🔮 Run Academic Risk & Performance Prediction", type="primary", use_container_width=True)

if run_prediction:
    row_features = [user_inputs[c] for c in feature_columns]
    input_df = pd.DataFrame([row_features], columns=feature_columns)
    scaled_input = scaler.transform(input_df)

    # Multi-task model forward pass
    gpa_scaled_pred, dropout_pred, perf_pred = model.predict(scaled_input, verbose=0)

    # Post-process targets
    gpa_pred_raw = y_scaler.inverse_transform(gpa_scaled_pred.reshape(-1, 1)).ravel()[0]
    future_gpa = float(np.clip(gpa_pred_raw, 0.0, 20.0))
    dropout_prob = float(dropout_pred.ravel()[0])
    perf_probs = perf_pred.ravel()
    perf_idx = int(np.argmax(perf_probs))
    perf_labels = ["Weak", "Average", "Good", "Excellent"]

    st.subheader("📊 Multi-Task Prediction & Assessment Results")
    res1, res2, res3 = st.columns(3)

    # 1. Future GPA Gauge
    with res1:
        gpa_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=future_gpa,
                number={"suffix": " / 20", "valueformat": ".2f"},
                title={"text": "<b>Predicted Future Term GPA</b>", "font": {"family": "Inter", "size": 16}},
                gauge={
                    "axis": {"range": [0, 20]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 10], "color": "#fee2e2"},
                        {"range": [10, 14], "color": "#fef9c3"},
                        {"range": [14, 20], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 3},
                        "thickness": 0.75,
                        "value": 10.0,
                    },
                },
            )
        )
        gpa_fig.update_layout(height=290, margin=dict(l=25, r=25, t=50, b=15))
        st.plotly_chart(gpa_fig, use_container_width=True)

    # 2. Dropout Risk Gauge
    with res2:
        is_high_risk = dropout_prob >= 0.5
        risk_color = "#dc2626" if is_high_risk else "#16a34a"
        risk_title = "High Risk ⚠️" if is_high_risk else "Low Risk ✅"

        risk_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=dropout_prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": f"<b>Dropout Risk: {risk_title}</b>", "font": {"family": "Inter", "size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 50], "color": "#dcfce7"},
                        {"range": [50, 100], "color": "#fee2e2"},
                    ],
                },
            )
        )
        risk_fig.update_layout(height=290, margin=dict(l=25, r=25, t=50, b=15))
        st.plotly_chart(risk_fig, use_container_width=True)

    # 3. Performance Level Probabilities
    with res3:
        perf_fig = go.Figure(
            go.Bar(
                x=perf_probs * 100,
                y=perf_labels,
                orientation="h",
                marker_color=["#ef4444", "#f97316", "#3b82f6", "#10b981"],
                text=[f"{p:.1%}" for p in perf_probs],
                textposition="outside",
            )
        )
        perf_fig.update_layout(
            title={"text": f"<b>Predicted Tier: {perf_labels[perf_idx]}</b>", "font": {"family": "Inter", "size": 16}},
            xaxis_title="Probability (%)",
            xaxis_range=[0, 115],
            height=290,
            margin=dict(l=25, r=25, t=50, b=15),
        )
        st.plotly_chart(perf_fig, use_container_width=True)

    # Advisory & Actionable Feedback
    if is_high_risk or future_gpa < 12.0:
        st.markdown(
            f"""
            <div class='danger-box'>
                <b>⚠️ Academic Risk Alert:</b> The model identifies an elevated probability of course dropout ({dropout_prob:.1%}) or significant academic decline.
                <br><b>Advisor Action Plan:</b> Refer the student to an academic advisor, establish a structured weekly tutoring schedule, monitor consecutive inactive days on LMS, and review assignment submission delays.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='recommendation-box'>
                <b>✅ Stable Academic Standing:</b> The student shows satisfactory engagement and progress, with an estimated Future GPA of <b>{future_gpa:.2f} / 20</b>.
                <br><b>Recommendation:</b> Maintain consistent study hours, encourage peer discussions on course forums, and promote advanced learning opportunities.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("💡 Adjust the student metrics above or select a preset profile from the sidebar, then click 'Run Academic Risk & Performance Prediction' to generate inferences.")
