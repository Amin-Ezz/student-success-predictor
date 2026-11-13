# 🎓 Student Academic GPA & Dropout Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning system designed to predict students' future Grade Point Average (GPA), assess their probability of dropping out, and classify their overall academic performance tier simultaneously using a **Multi-Task Deep Neural Network (MT-DNN)**.

Featuring an interactive **Streamlit** dashboard ready for one-click deployment on **Streamlit Community Cloud** or local execution.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Multi-Task Model Architecture](#multi-task-model-architecture)
- [Dataset Characteristics](#dataset-characteristics)
- [Model Performance](#model-performance)
- [Project Directory Structure](#project-directory-structure)
- [Getting Started Locally](#getting-started-locally)
- [Deployment on Streamlit Community Cloud](#deployment-on-streamlit-community-cloud)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## 📖 Overview

Student retention and timely academic intervention are critical challenges in higher education and modern e-learning platforms. Early identification of academic distress allows educators and advisors to provide tailored support before a student drops out.

This repository provides:
1. **Synthetic Data Generation Engine**: Creates realistic datasets combining demographic data, academic histories, and detailed Learning Management System (LMS) engagement logs.
2. **Multi-Output Deep Learning Pipeline**: A shared neural network backbone that branches into three specialized prediction heads.
3. **Production Web Application**: A bilingual-friendly, RTL-styled Streamlit interface with interactive Plotly visual gauges, preset student profiles, and actionable advising recommendations.

---

## ✨ Key Features

- **Simultaneous Multi-Target Prediction**:
  - **Future GPA**: Continuous regression score on a 0–20 scale.
  - **Dropout Risk**: Probability estimate with automated risk category tagging (Low vs. High Risk).
  - **Academic Performance Level**: 4-class categorization (`Weak`, `Average`, `Good`, `Excellent`).
- **Interactive Web Interface**:
  - Quick-test student presets (*High Achiever*, *At-Risk Student*, *Average Student*).
  - Categorized input tabs (*Personal & Family Background*, *Academic Records*, *Online LMS Activity*).
  - Real-time gauge meters and probability bar charts.
- **Production Ready for Cloud**:
  - Pre-trained model artifacts and scalers bundled in `artifacts/` for near-instant boot on Streamlit Cloud without retraining overhead.
  - Modular Python package structure adhering to clean software engineering practices.

---

## 🧠 Multi-Task Model Architecture

Rather than training three disjoint models, this project employs **Multi-Task Learning (MTL)**. A shared representation backbone learns general latent features of student engagement and achievement, which regularizes the network and improves generalization.

```
                           [ 36 Input Features ]
                                     │
                     ┌───────────────▼──────────────┐
                     │ Dense(256) + BatchNorm + D(0.3)│
                     │ Dense(128) + BatchNorm + D(0.25)│
                     │  Dense(64) + BatchNorm + D(0.2) │
                     │          Dense(32)             │
                     └───────────────┬──────────────┘
                                     │ (Shared Latent Space)
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
  ┌─────────▼────────┐     ┌─────────▼────────┐     ┌─────────▼────────┐
  │  Future GPA Head │     │ Dropout Risk Head│     │ Performance Head │
  │  Dense(32) -> 16 │     │  Dense(32) -> 16 │     │  Dense(32) -> 16 │
  │  Dense(1, Linear)│     │ Dense(1, Sigmoid)│     │ Dense(4, Softmax)│
  └──────────────────┘     └──────────────────┘     └──────────────────┘
```

- **Loss Functions**:
  - Future GPA: Mean Squared Error (`MSE`), weight = `1.0`
  - Dropout Risk: Binary Crossentropy (`BCE`), weight = `1.0`
  - Performance Level: Sparse Categorical Crossentropy (`SCCE`), weight = `1.5`
- **Optimizer**: Adam with learning rate reduction on plateau (`ReduceLROnPlateau`) and `EarlyStopping`.

---

## 📊 Dataset Characteristics

The data incorporates 36 features across three domains:

| Category | Key Features Included |
| :--- | :--- |
| **Personal & Demographics** | Age, Gender, Department, Semester, Family Income, Parent Education Level, Scholarship Status |
| **Academic Performance** | High School GPA, Current GPA, Weekly Study Hours, Sleep Hours, Attendance Rate, Assignment Average, Quiz Average, Midterm Score, Class Participation, Previous Course Failures |
| **Online LMS Engagement** | Login Frequency, Total Online Time, Lecture Videos Watched, Course Completion Rate, Practice Exam Scores, Forum Posts, Questions Asked/Answered, Assignment Delays, Consecutive Inactive Days, Device Usage (Mobile vs. Desktop), AI Tools Usage Frequency |

---

## 📈 Model Performance

Evaluated on an independent 20% holdout test set (1,000 samples):

| Task | Metric | Value |
| :--- | :--- | :--- |
| **Task 1: Future GPA** | $R^2$ Score | **93.2%** |
| | Mean Absolute Error (MAE) | **0.23** |
| | Root Mean Squared Error (RMSE) | **0.28** |
| **Task 2: Dropout Risk** | Accuracy | **90.7%** |
| | Precision | **91.6%** |
| | Recall | **92.9%** |
| | F1-Score | **92.2%** |
| **Task 3: Performance Tier** | Accuracy | **92.5%** |
| | Weighted Precision | **92.6%** |
| | Weighted Recall | **92.5%** |
| | Weighted F1-Score | **92.5%** |

---

## 📁 Project Directory Structure

```text
predicting-student-gpa-dropout-risk/
│
├── .streamlit/
│   └── config.toml               # Streamlit theme & server configuration
│
├── artifacts/                    # Pre-trained models and metadata for Cloud deployment
│   ├── model.keras               # Saved Keras multi-task neural network
│   ├── scaler.pkl                # Fitted StandardScaler for input features
│   ├── y_scaler.pkl              # Fitted StandardScaler for continuous GPA
│   ├── feature_columns.json      # Ordered feature column names
│   ├── feature_stats.json        # Min/mean/max bounds for UI controls
│   └── test_metrics.json         # Evaluated metrics on the test dataset
│
├── data/
│   ├── raw/                      # Raw simulated academic & online LMS datasets
│   └── processed/                # Merged and imputed dataset (CSV)
│
├── notebooks/
│   └── student_performance_modeling.ipynb  # Interactive exploratory & training notebook
│
├── src/                          # Modular Python source code
│   ├── __init__.py
│   ├── data_generator.py         # Synthetic student dataset generator
│   ├── preprocess.py             # Feature scaling, imputation, and train/test splitting
│   ├── model.py                  # Keras Multi-Task architecture definition
│   ├── train.py                  # End-to-end training and artifact export pipeline
│   ├── evaluate.py               # Comprehensive metric computation utilities
│   └── predict.py                # Reusable student prediction inference class
│
├── app.py                        # Streamlit web dashboard entrypoint
├── requirements.txt              # Production dependency specifications
├── .gitignore                    # Git exclusion rules
├── LICENSE                       # MIT Open Source License
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/predicting-student-gpa-dropout-risk.git
cd predicting-student-gpa-dropout-risk
```

### 2. Set Up a Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train the Model & Re-generate Data
The repository already includes pre-trained artifacts in `artifacts/`. If you wish to retrain from scratch:
```bash
python -m src.train
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This project is tailored specifically for smooth, zero-config deployment on **[Streamlit Community Cloud](https://streamlit.io/cloud)**:

### Step 1: Push Your Code to GitHub
Ensure all files (including the `artifacts/` folder) are committed:
```bash
git init
git add .
git commit -m "Initial commit: Student GPA & Dropout Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

### Step 2: Connect to Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `YOUR_USERNAME/YOUR_REPOSITORY_NAME`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Click **"Deploy!"**.

> 💡 **Why deployment is instantaneous:**
> Because the trained neural network model and metadata are pre-packaged in the `artifacts/` directory, Streamlit Cloud will launch directly without requiring GPU/CPU training cycles or risking memory limit timeouts!

---

## 🛠️ Technologies Used

- **Application & Dashboard**: [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/)
- **Data Science & ML**: [Scikit-Learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Joblib](https://joblib.readthedocs.io/)
- **Visualization**: [Plotly Graph Objects](https://plotly.com/python/graph-objects/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
