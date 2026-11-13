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
# Page Configuration & Professional Styling
# ==========================================================
st.set_page_config(
    page_title="سامانه پیش‌بینی وضعیت تحصیلی و ریسک انصراف دانشجو",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* RTL Layout for Persian content */
    .rtl-block {
        direction: rtl;
        text-align: right;
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

    /* Alert / Recommendation Box */
    .recommendation-box {
        background-color: #eff6ff;
        border-right: 4px solid #2563eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 15px;
        color: #1e3a8a;
        direction: rtl;
        text-align: right;
    }
    
    .danger-box {
        background-color: #fef2f2;
        border-right: 4px solid #dc2626;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 15px;
        color: #991b1b;
        direction: rtl;
        text-align: right;
    }

    [data-testid="stMetricLabel"] {
        direction: rtl;
        text-align: right;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        direction: ltr;
        text-align: right;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


@st.cache_resource(show_spinner="در حال بارگذاری مدل یادگیری عمیق...")
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
        "⚠️ فایل‌های مدل و آرتیفکت‌ها یافت نشدند.\n\n"
        "برای تولید آرتیفکت‌ها، لطفاً دستور زیر را در ترمینال اجرا کنید:\n\n"
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
# Sidebar: Presets & Model Info
# ==========================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=600&q=80",
        caption="سیستم پایش هوشمند وضعیت تحصیلی دانشجویان",
        use_container_width=True,
    )
    st.markdown("### 🎯 پروفایل‌های نمونه (تست سریع)")
    st.caption("با انتخاب هر پروفایل، ورودی‌ها به صورت خودکار مقادیر فرضی می‌گیرند:")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("🌟 دانشجوی ممتاز", use_container_width=True):
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
        if st.button("⚠️ دانشجوی در خطر", use_container_width=True):
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

    if st.button("🔄 بازنشانی به میانگین‌ها", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith(("p1_", "p2_", "p3_")):
                del st.session_state[k]
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ درباره پروژه")
    st.info(
        "این سامانه از یک شبکه عصبی عمیق چندخروجی (Multi-Output DNN) برای پیش‌بینی هم‌زمان "
        "سه هدف (معدل آینده، خطر ترک تحصیل، و سطح علمی) بهره می‌برد.\n\n"
        "**توسعه‌داده‌شده برای ارائه پورتفولیو و استقرار روی Streamlit Cloud.**"
    )

# ==========================================================
# Main Header & Model Performance Metrics
# ==========================================================
st.markdown("<div class='rtl-block'>", unsafe_allow_html=True)
st.title("🎓 سامانه هوشمند پیش‌بینی وضعیت تحصیلی و ریسک انصراف دانشجو")
st.markdown(
    "با ورود مشخصات آموزشی، فردی و فعالیت در سامانه مجازی، مدل عمیق وضعیت ترم آتی دانشجو را پیش‌بینی می‌کند."
)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📈 مشاهده کارایی و دقت مدل روی مجموعه داده آزمایشی (Test Set)", expanded=False):
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(
            label="ضریب تعیین معدل آینده (R² Score)",
            value=f"{test_metrics['future_gpa']['r2']:.1%}",
            delta=f"MAE: {test_metrics['future_gpa']['mae']:.2f}",
        )
    with m_col2:
        st.metric(
            label="دقت پیش‌بینی انصراف (Dropout Accuracy)",
            value=f"{test_metrics['dropout']['accuracy']:.1%}",
            delta=f"F1: {test_metrics['dropout']['f1']:.1%}",
        )
    with m_col3:
        st.metric(
            label="دقت طبقه‌بندی سطح عملکرد (Academic Tier)",
            value=f"{test_metrics['performance']['accuracy']:.1%}",
            delta=f"F1: {test_metrics['performance']['f1']:.1%}",
        )

st.divider()

# ==========================================================
# Input Forms inside Tabs
# ==========================================================
tab_personal, tab_academic, tab_online = st.tabs([
    "👤 اطلاعات فردی و خانوادگی",
    "📚 عملکرد تحصیلی و سوابق",
    "💻 فعالیت در سامانه یادگیری آنلاین (LMS)",
])

user_inputs = {}

with tab_personal:
    st.markdown("<div class='rtl-block'><h5>مشخصات فردی و اقتصادی</h5></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["Age"] = make_slider("Age", "سن دانشجو", "p1")
        user_inputs["Gender"] = st.selectbox(
            "جنسیت",
            [0, 1],
            format_func=lambda x: "مرد" if x == 1 else "زن",
            key="p1_gender",
        )
        user_inputs["Department"] = st.selectbox(
            "رشته / دپارتمان",
            [0, 1, 2, 3, 4],
            format_func=lambda x: f"دپارتمان مهندسی / علوم {x + 1}",
            key="p1_dept",
        )
        user_inputs["Semester"] = make_slider("Semester", "ترم تحصیلی جاری", "p1")

    with c2:
        user_inputs["FamilyIncome"] = make_slider(
            "FamilyIncome", "درآمد تخمینی خانواده (میلیون تومان)", "p1"
        )
        user_inputs["ParentEducation"] = st.selectbox(
            "میزان تحصیلات والدین",
            [0, 1, 2, 3],
            format_func=lambda x: ["دیپلم یا کمتر", "کاردانی", "کارشناسی", "کارشناسی ارشد و دکتری"][x],
            index=2,
            key="p1_pedu",
        )
        user_inputs["Scholarship"] = st.selectbox(
            "وضعیت بورسیه تحصیلی",
            [0, 1],
            format_func=lambda x: "ندارد" if x == 0 else "دارای بورسیه",
            key="p1_schol",
        )

with tab_academic:
    st.markdown("<div class='rtl-block'><h5>نمرات و سوابق درسی</h5></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["HighSchoolGPA"] = make_slider("HighSchoolGPA", "معدل دیپلم (از ۲۰)", "p2")
        user_inputs["CurrentGPA"] = make_slider("CurrentGPA", "معدل کل فعلی (از ۲۰)", "p2")
        user_inputs["StudyHoursPerWeek"] = make_slider("StudyHoursPerWeek", "ساعت مطالعه هفتگی", "p2")
        user_inputs["SleepHours"] = make_slider("SleepHours", "میانگین ساعت خواب در شبانه‌روز", "p2")
        user_inputs["AttendanceRate"] = make_slider("AttendanceRate", "نرخ حضور در کلاس‌ها (درصد)", "p2")
        user_inputs["LibraryVisits"] = make_slider("LibraryVisits", "دفعات مراجعه به کتابخانه در ماه", "p2")

    with c2:
        user_inputs["AssignmentAverage"] = make_slider("AssignmentAverage", "میانگین نمرات تکالیف", "p2")
        user_inputs["QuizAverage"] = make_slider("QuizAverage", "میانگین نمرات کوییزها", "p2")
        user_inputs["MidtermScore"] = make_slider("MidtermScore", "نمره آزمون میان‌ترم", "p2")
        user_inputs["ClassParticipation"] = make_slider("ClassParticipation", "نمره مشارکت کلاسی", "p2")
        user_inputs["PreviousFailures"] = make_slider("PreviousFailures", "تعداد دروس افتاده قبلی", "p2")
        user_inputs["InternetUsage"] = make_slider("InternetUsage", "ساعت استفاده روزانه از اینترنت (غیردرسی)", "p2")

with tab_online:
    st.markdown("<div class='rtl-block'><h5>رفتار و فعالیت در سامانه LMS</h5></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["LoginFrequency"] = make_slider("LoginFrequency", "تعداد ورود به سامانه در هفته", "p3")
        user_inputs["TotalLoginTime"] = make_slider("TotalLoginTime", "مدت زمان حضور در سامانه (دقیقه/هفته)", "p3")
        user_inputs["VideosWatched"] = make_slider("VideosWatched", "تعداد ویدیوهای آموزشی دیده‌شده", "p3")
        user_inputs["CompletionRate"] = make_slider("CompletionRate", "درصد تکمیل سرفصل‌های دوره", "p3")
        user_inputs["AveragePracticeScore"] = make_slider("AveragePracticeScore", "میانگین نمره تمرین‌های آنلاین", "p3")
        user_inputs["PracticeTestsTaken"] = make_slider("PracticeTestsTaken", "تعداد کوییزهای تمرینی انجام شده", "p3")
        user_inputs["DownloadedResources"] = make_slider("DownloadedResources", "تعداد جزوات و فایل‌های دانلود شده", "p3")
        user_inputs["AssignmentSubmissionDelay"] = make_slider("AssignmentSubmissionDelay", "میانگین تاخیر در تحویل تکالیف (روز)", "p3")
        user_inputs["ConsecutiveInactiveDays"] = make_slider("ConsecutiveInactiveDays", "تعداد روزهای متوالی عدم حضور در سامانه", "p3")

    with c2:
        user_inputs["ForumPosts"] = make_slider("ForumPosts", "تعداد پست‌های ارسالی در تالار گفتگو", "p3")
        user_inputs["QuestionsAsked"] = make_slider("QuestionsAsked", "تعداد سوالات پرسیده شده", "p3")
        user_inputs["QuestionsAnswered"] = make_slider("QuestionsAnswered", "تعداد پاسخ‌های داده شده به دیگران", "p3")
        user_inputs["MobileUsage"] = make_slider("MobileUsage", "درصد ورود با تلفن همراه", "p3")
        user_inputs["DesktopUsage"] = make_slider("DesktopUsage", "درصد ورود با رایانه", "p3")
        user_inputs["NightStudyPercentage"] = make_slider("NightStudyPercentage", "درصد مطالعه در ساعات شب (۱۲ تا ۶ صبح)", "p3")
        user_inputs["WeekendStudyPercentage"] = make_slider("WeekendStudyPercentage", "درصد مطالعه در روزهای تعطیل", "p3")
        user_inputs["AIUsageFrequency"] = make_slider("AIUsageFrequency", "میزان استفاده از ابزارهای هوش مصنوعی", "p3")
        user_inputs["CourseCompletionTime"] = make_slider("CourseCompletionTime", "زمان تخمینی اتمام دوره (هفته)", "p3")
        user_inputs["CertificateEarned"] = st.selectbox(
            "دریافت گواهی پایان دوره",
            [0, 1],
            format_func=lambda x: "خیر" if x == 0 else "بله",
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
    run_prediction = st.button("🔮 اجرای پیش‌بینی وضعیت دانشجو", type="primary", use_container_width=True)

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
    perf_labels = ["ضعیف", "متوسط", "خوب", "عالی"]

    st.markdown("<div class='rtl-block'><h3>📊 نتایج تحلیل و پیش‌بینی هوشمند</h3></div>", unsafe_allow_html=True)
    res1, res2, res3 = st.columns(3)

    # 1. Future GPA Gauge
    with res1:
        gpa_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=future_gpa,
                number={"suffix": " از ۲۰", "valueformat": ".2f"},
                title={"text": "<b>معدل پیش‌بینی‌شده ترم آتی</b>", "font": {"family": "Vazirmatn", "size": 16}},
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
        risk_title = "ریسک بالا ⚠️" if is_high_risk else "ریسک پایین ✅"

        risk_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=dropout_prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": f"<b>خطر انصراف از تحصیل: {risk_title}</b>", "font": {"family": "Vazirmatn", "size": 16}},
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
            title={"text": f"<b>سطح علمی تخمینی: {perf_labels[perf_idx]}</b>", "font": {"family": "Vazirmatn", "size": 16}},
            xaxis_title="احتمال وقوع (٪)",
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
                <b>⚠️ هشدار تحصیلی:</b> دانشجو دارای ریسک انصراف ({dropout_prob:.1%}) یا افت معدل است.
                <br><b>اقدامات پیشنهادی مشاور:</b> ارجاع به استاد راهنما، تنظیم برنامه مطالعه هدفمند، افزایش ساعات حضور در سامانه آموزشی، و بررسی عدم فعالیت‌های متوالی.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='recommendation-box'>
                <b>✅ وضعیت پایدار:</b> وضعیت تحصیلی دانشجو مطلوب ارزیابی شد. معدل پیش‌بینی‌شده: {future_gpa:.2f} از ۲۰.
                <br><b>پیشنهاد:</b> ادامه روند مطالعه فعلی و مشارکت مستمر در تالارهای گفتگو و پروژه‌های تکمیلی.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("💡 برای مشاهده نتایج پیش‌بینی، مقادیر فرم بالا را تنظیم کرده یا از دکمه‌های پروفایل در ستون کناری استفاده کرده و بر روی «اجرای پیش‌بینی وضعیت دانشجو» کلیک کنید.")
