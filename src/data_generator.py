"""
Data Generator Module for Academic and Online Learning Datasets.
Simulates realistic student demographic, academic performance, and online LMS behavior.
"""

import os
import numpy as np
import pandas as pd


def generate_academic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic academic records for students."""
    np.random.seed(seed)

    student_id = np.arange(1, n_samples + 1)
    age = np.random.randint(18, 30, n_samples)
    gender = np.random.randint(0, 2, n_samples)
    department = np.random.choice(
        [0, 1, 2, 3, 4],
        size=n_samples,
        p=[0.22, 0.20, 0.18, 0.20, 0.20],
    )
    semester = np.random.randint(1, 9, n_samples)

    family_income = np.clip(np.random.normal(60, 18, n_samples), 20, 120)
    parent_education = np.random.choice(
        [0, 1, 2, 3],
        size=n_samples,
        p=[0.18, 0.32, 0.34, 0.16],
    )
    scholarship = (family_income < 50).astype(int)

    high_school_gpa = np.clip(
        np.random.normal(15.5 + parent_education * 0.4, 1.8, n_samples),
        8,
        20,
    )
    study_hours_per_week = np.clip(
        np.random.normal(18 + (high_school_gpa - 15), 5, n_samples),
        5,
        40,
    )
    sleep_hours = np.clip(
        np.random.normal(7.2 - (study_hours_per_week - 18) * 0.04, 0.7, n_samples),
        4,
        9,
    )
    internet_usage = np.clip(
        np.random.normal(4.5 - (study_hours_per_week - 18) * 0.08, 1, n_samples),
        1,
        8,
    )
    library_visits = np.clip(
        np.random.normal(study_hours_per_week / 3, 2, n_samples),
        0,
        20,
    ).round()
    class_participation = np.clip(
        np.random.normal(60 + study_hours_per_week, 10, n_samples),
        20,
        100,
    )
    attendance_rate = np.clip(
        np.random.normal(
            72 + (study_hours_per_week * 0.8) - (internet_usage * 2),
            6,
            n_samples,
        ),
        40,
        100,
    )
    assignment_avg = np.clip(
        np.random.normal(
            45 + high_school_gpa * 2.2 + study_hours_per_week * 0.6,
            6,
            n_samples,
        ),
        40,
        100,
    )
    quiz_avg = np.clip(
        np.random.normal(assignment_avg * 0.90, 5, n_samples),
        35,
        100,
    )
    midterm_score = np.clip(
        np.random.normal((assignment_avg + quiz_avg + attendance_rate) / 3, 5, n_samples),
        35,
        100,
    )
    current_gpa = np.clip(
        np.random.normal(
            (
                high_school_gpa * 0.45
                + (assignment_avg / 5) * 0.20
                + (quiz_avg / 5) * 0.20
                + (midterm_score / 5) * 0.15
            ),
            0.5,
            n_samples,
        ),
        8,
        20,
    )
    previous_failures = np.clip(
        ((16 - current_gpa) / 2 + np.random.normal(0, 0.5, n_samples)).round(),
        0,
        6,
    ).astype(int)

    df_academic = pd.DataFrame({
        "StudentID": student_id,
        "Age": age,
        "Gender": gender,
        "Department": department,
        "Semester": semester,
        "FamilyIncome": family_income.round(2),
        "ParentEducation": parent_education,
        "Scholarship": scholarship,
        "HighSchoolGPA": high_school_gpa.round(2),
        "CurrentGPA": current_gpa.round(2),
        "StudyHoursPerWeek": study_hours_per_week.round(1),
        "SleepHours": sleep_hours.round(1),
        "InternetUsage": internet_usage.round(1),
        "LibraryVisits": library_visits.astype(int),
        "AttendanceRate": attendance_rate.round(1),
        "AssignmentAverage": assignment_avg.round(1),
        "QuizAverage": quiz_avg.round(1),
        "MidtermScore": midterm_score.round(1),
        "ClassParticipation": class_participation.round(1),
        "PreviousFailures": previous_failures,
    })

    # Introduce 20% missing values
    for col in ["HighSchoolGPA", "QuizAverage", "MidtermScore", "FamilyIncome"]:
        mask = df_academic.sample(frac=0.2, random_state=seed).index
        df_academic.loc[mask, col] = np.nan

    return df_academic


def generate_online_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic LMS and online learning behavior records."""
    np.random.seed(seed)

    student_id = np.arange(1, n_samples + 1)
    engagement = np.clip(np.random.normal(70, 15, n_samples), 20, 100)

    login_freq = np.clip(np.random.normal(engagement / 3, 3, n_samples), 2, 40).round()
    total_login_time = np.clip(np.random.normal(login_freq * 2.5, 10, n_samples), 20, 180)
    videos_watched = np.clip(np.random.normal(login_freq * 3, 8, n_samples), 5, 120).round()
    completion_rate = np.clip(np.random.normal(engagement, 8, n_samples), 20, 100)
    downloaded_resources = np.clip(
        np.random.normal(videos_watched * 0.55, 4, n_samples), 0, 60
    ).round()
    practice_tests = np.clip(np.random.normal(videos_watched / 5, 2, n_samples), 0, 25).round()
    avg_practice_score = np.clip(
        np.random.normal(completion_rate * 0.90, 6, n_samples), 30, 100
    )

    forum_posts = np.clip(np.random.normal(engagement / 10, 2, n_samples), 0, 20).round()
    questions_asked = np.clip(np.random.normal(forum_posts * 0.8, 2, n_samples), 0, 20).round()
    questions_answered = np.clip(np.random.normal(forum_posts * 0.9, 2, n_samples), 0, 20).round()

    delay = np.clip(np.random.normal(15 - engagement / 8, 2, n_samples), 0, 15)
    inactive_days = np.clip(np.random.normal(12 - engagement / 10, 2, n_samples), 0, 15).round()
    course_time = np.clip(np.random.normal(18 - engagement / 8, 3, n_samples), 5, 20)

    mobile_usage = np.clip(np.random.normal(65, 18, n_samples), 10, 100)
    desktop_usage = np.clip(
        100 - mobile_usage + np.random.normal(0, 5, n_samples), 0, 100
    )
    night_study = np.clip(np.random.normal(35, 15, n_samples), 0, 100)
    weekend_study = np.clip(np.random.normal(30, 12, n_samples), 0, 100)
    ai_usage = np.clip(np.random.normal(engagement / 10, 2, n_samples), 0, 15).round()

    cert_earned = (
        (completion_rate > 80) & (avg_practice_score > 70)
    ).astype(int)

    df_online = pd.DataFrame({
        "StudentID": student_id,
        "LoginFrequency": login_freq.astype(int),
        "TotalLoginTime": total_login_time.round(1),
        "VideosWatched": videos_watched.astype(int),
        "CompletionRate": completion_rate.round(1),
        "ForumPosts": forum_posts.astype(int),
        "QuestionsAsked": questions_asked.astype(int),
        "QuestionsAnswered": questions_answered.astype(int),
        "AssignmentSubmissionDelay": delay.round(1),
        "DownloadedResources": downloaded_resources.astype(int),
        "PracticeTestsTaken": practice_tests.astype(int),
        "AveragePracticeScore": avg_practice_score.round(1),
        "ConsecutiveInactiveDays": inactive_days.astype(int),
        "MobileUsage": mobile_usage.round(1),
        "DesktopUsage": desktop_usage.round(1),
        "NightStudyPercentage": night_study.round(1),
        "WeekendStudyPercentage": weekend_study.round(1),
        "AIUsageFrequency": ai_usage.astype(int),
        "CourseCompletionTime": course_time.round(1),
        "CertificateEarned": cert_earned,
    })

    # Introduce 20% missing values
    for col in [
        "LoginFrequency",
        "ConsecutiveInactiveDays",
        "DownloadedResources",
        "AIUsageFrequency",
    ]:
        mask = df_online.sample(frac=0.2, random_state=seed).index
        df_online.loc[mask, col] = np.nan

    return df_online


def build_full_dataset(
    n_samples: int = 5000,
    seed: int = 42,
    save_raw: bool = True,
    save_processed: bool = True,
    data_dir: str = "data",
) -> pd.DataFrame:
    """Generate, impute, compute targets, and return consolidated DataFrame."""
    np.random.seed(seed)

    raw_academic = generate_academic_data(n_samples=n_samples, seed=seed)
    raw_online = generate_online_data(n_samples=n_samples, seed=seed)

    if save_raw:
        os.makedirs(f"{data_dir}/raw", exist_ok=True)
        raw_academic.to_csv(f"{data_dir}/raw/academic_data_raw.csv", index=False)
        raw_online.to_csv(f"{data_dir}/raw/online_data_raw.csv", index=False)

    # Impute missing values with mean
    for col in ["HighSchoolGPA", "QuizAverage", "MidtermScore", "FamilyIncome"]:
        raw_academic[col] = raw_academic[col].fillna(raw_academic[col].mean())

    for col in [
        "LoginFrequency",
        "ConsecutiveInactiveDays",
        "DownloadedResources",
        "AIUsageFrequency",
    ]:
        raw_online[col] = raw_online[col].fillna(raw_online[col].mean())

    # Merge online_df and academic_df (drop redundant StudentID in academic)
    df = pd.concat([raw_online, raw_academic.drop(columns="StudentID")], axis=1)

    # Target 1: FutureGPA (Regression)
    noise_gpa = np.random.normal(0, 0.22, len(df))
    df["FutureGPA"] = (
        0.40 * df["CurrentGPA"]
        + 0.15 * (df["AssignmentAverage"] / 5)
        + 0.10 * (df["QuizAverage"] / 5)
        + 0.08 * (df["MidtermScore"] / 5)
        + 0.08 * (df["AttendanceRate"] / 5)
        + 0.05 * (df["StudyHoursPerWeek"] / 2)
        + 0.05 * (df["AveragePracticeScore"] / 5)
        + 0.04 * (df["CompletionRate"] / 5)
        + 0.03 * (df["LoginFrequency"] / 2)
        + 0.02 * df["LibraryVisits"]
        - 0.18 * df["PreviousFailures"]
        - 0.10 * df["ConsecutiveInactiveDays"]
        - 0.06 * df["AssignmentSubmissionDelay"]
        - 0.04 * df["InternetUsage"]
        + noise_gpa
    )
    df["FutureGPA"] = np.clip(df["FutureGPA"], 0, 20).round(2)

    # Target 2: DropoutRisk (Binary Classification)
    noise_risk = np.random.normal(0, 0.2, len(df))
    risk_score = (
        0.18 * df["PreviousFailures"]
        + 0.12 * df["ConsecutiveInactiveDays"]
        + 0.10 * df["AssignmentSubmissionDelay"]
        + 0.08 * (100 - df["AttendanceRate"])
        + 0.07 * (100 - df["CompletionRate"])
        + 0.06 * (100 - df["AveragePracticeScore"])
        + 0.05 * (40 - df["StudyHoursPerWeek"])
        + 0.04 * (20 - df["CurrentGPA"])
        + 0.03 * (25 - df["LoginFrequency"])
        - 0.03 * df["Scholarship"]
        + noise_risk
    )
    prob_dropout = 1 / (1 + np.exp(-(risk_score - 8) / 0.35))
    df["DropoutRisk"] = (np.random.rand(len(df)) < prob_dropout).astype(int)

    # Target 3: PerformanceLevel (Multi-Class Classification: 0: Weak, 1: Average, 2: Good, 3: Excellent)
    noise_perf = np.random.normal(0, 0.8, len(df))
    perf_score = (
        4.0 * df["CurrentGPA"]
        + 0.35 * df["AssignmentAverage"]
        + 0.30 * df["QuizAverage"]
        + 0.30 * df["MidtermScore"]
        + 0.20 * df["AttendanceRate"]
        + 0.15 * df["AveragePracticeScore"]
        + 0.12 * df["CompletionRate"]
        + 0.08 * df["StudyHoursPerWeek"]
        + 0.05 * df["LoginFrequency"]
        + 0.03 * df["LibraryVisits"]
        - 2.50 * df["PreviousFailures"]
        - 1.50 * df["ConsecutiveInactiveDays"]
        - 1.20 * df["AssignmentSubmissionDelay"]
        + noise_perf
    )
    q_edges = perf_score.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy().copy()
    q_edges[0] = -np.inf
    q_edges[-1] = np.inf

    perf_cat = pd.cut(
        perf_score,
        bins=q_edges,
        labels=["Weak", "Average", "Good", "Excellent"],
        include_lowest=True,
    )
    df["PerformanceLevel"] = perf_cat.map({
        "Weak": 0,
        "Average": 1,
        "Good": 2,
        "Excellent": 3,
    }).astype(int)

    if save_processed:
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
        df.to_csv(f"{data_dir}/processed/students_combined_processed.csv", index=False)

    return df


if __name__ == "__main__":
    print("Generating student dataset...")
    data = build_full_dataset()
    print(f"Dataset generated successfully! Shape: {data.shape}")
    print(f"Target distributions:\n - FutureGPA mean: {data['FutureGPA'].mean():.2f}")
    print(f" - DropoutRisk rate: {data['DropoutRisk'].mean():.1%}")
    print(f" - PerformanceLevel distribution:\n{data['PerformanceLevel'].value_counts().sort_index()}")
