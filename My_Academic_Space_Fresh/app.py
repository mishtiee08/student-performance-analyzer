import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# MY ACADEMIC SPACE - SELF-CONTAINED STREAMLIT APPLICATION
# =========================================================

st.set_page_config(
    page_title="My Academic Space",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ASSESSMENTS_FILE = DATA_DIR / "assessments.csv"
GRADES_FILE = DATA_DIR / "semester_grades.csv"
PLANNER_FILE = DATA_DIR / "study_planner.csv"
EXAMS_FILE = DATA_DIR / "exams.csv"
MODEL_FILE = MODEL_DIR / "student_model.pkl"

ASSESSMENT_COLUMNS = [
    "Date", "Subject", "Assessment", "Marks", "Total Marks", "Percentage"
]
GRADE_COLUMNS = ["Semester", "Subject", "Credits", "Grade Point"]
PLANNER_COLUMNS = ["Date", "Time", "Subject", "Task", "Duration", "Completed"]
EXAM_COLUMNS = ["Exam", "Subject", "Exam Date", "Notes"]


def load_csv(path, columns):
    if path.exists():
        try:
            df = pd.read_csv(path)
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            return df[columns]
        except Exception:
            st.warning(f"Could not read {path.name}. A fresh data table will be created.")
    return pd.DataFrame(columns=columns)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_assessments():
    return load_csv(ASSESSMENTS_FILE, ASSESSMENT_COLUMNS)


def load_grades():
    return load_csv(GRADES_FILE, GRADE_COLUMNS)


def load_planner():
    return load_csv(PLANNER_FILE, PLANNER_COLUMNS)


def load_exams():
    return load_csv(EXAMS_FILE, EXAM_COLUMNS)


def sgpa_for(grades, semester):
    data = grades[grades["Semester"].astype(str) == str(semester)].copy()
    if data.empty:
        return 0.0
    credits = pd.to_numeric(data["Credits"], errors="coerce").fillna(0)
    points = pd.to_numeric(data["Grade Point"], errors="coerce").fillna(0)
    total = credits.sum()
    return round((credits * points).sum() / total, 2) if total else 0.0


def cgpa_for(grades):
    if grades.empty:
        return 0.0
    credits = pd.to_numeric(grades["Credits"], errors="coerce").fillna(0)
    points = pd.to_numeric(grades["Grade Point"], errors="coerce").fillna(0)
    total = credits.sum()
    return round((credits * points).sum() / total, 2) if total else 0.0


def ensure_model():
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    # Small illustrative training set for a project demo.
    # Replace with a real, appropriately collected dataset before making claims.
    sample = pd.DataFrame({
        "attendance": [45, 55, 60, 65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 96, 98],
        "study_hours": [1, 1, 2, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 6, 7],
        "assignment_marks": [35, 45, 50, 55, 60, 58, 65, 68, 70, 75, 78, 82, 85, 88, 92, 96],
        "internal_marks": [30, 40, 45, 50, 55, 52, 60, 62, 68, 70, 75, 78, 82, 86, 90, 95],
        "previous_cgpa": [4.0, 4.8, 5.0, 5.5, 6.0, 5.8, 6.3, 6.5, 7.0, 7.2, 7.5, 7.8, 8.0, 8.4, 8.8, 9.2],
        "performance": [32, 40, 45, 48, 54, 52, 60, 64, 68, 72, 76, 80, 84, 88, 93, 97]
    })
    features = [
        "attendance", "study_hours", "assignment_marks",
        "internal_marks", "previous_cgpa"
    ]
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(sample[features], sample["performance"])
    joblib.dump(model, MODEL_FILE)
    return model


# ----------------------- STYLE -----------------------


st.markdown("""
<style>
.stApp {
    background: #241832;
    color: #F7F1FF;
}

[data-testid="stSidebar"] {
    background: #352449;
}

h1, h2, h3, p, label {
    color: #F7F1FF;
}

div[data-testid="stMetric"] {
    background: #3B2A50;
    border: 1px solid #7952A0;
    padding: 16px;
    border-radius: 14px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #9166BB;
    color: white;
    border: 0;
    border-radius: 10px;
}

.stButton > button:hover {
    background: #A982D0;
    color: white;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("🎓 SPA")
    st.caption("My Academic Space")
    st.divider()
    st.write("Your personal academic dashboard")
    st.write("📚 Study planner")
    st.write("🎯 Exam countdown")
    st.write("📊 Performance tracking")
    st.write("🧮 CGPA & SGPA")
    st.divider()
    st.caption("Your data is stored locally in this project folder.")

# Native Streamlit banner: intentionally no HTML banner.
st.markdown(" ")
with st.container(border=True):
    st.caption("YOUR PERSONAL ACADEMIC SPACE")
    st.title("Make progress, not pressure.")
    st.write("Your studies, your achievements, your journey — all in one place. 🌷")

assessments = load_assessments()
grades = load_grades()
planner = load_planner()
exams = load_exams()

tabs = st.tabs([
    "🏠 Dashboard",
    "🎓 CGPA & SGPA",
    "📝 Assessments",
    "📊 My Progress",
    "🤖 AI Predictor",
    "📅 Study Planner",
    "⏳ Exam Countdown"
])

# ======================= DASHBOARD =======================
with tabs[0]:
    st.subheader("Your Academic Overview")
    percentages = pd.to_numeric(assessments["Percentage"], errors="coerce").dropna()
    avg = percentages.mean() if not percentages.empty else 0
    best = percentages.max() if not percentages.empty else 0
    cgpa = cgpa_for(grades)
    pending = 0
    if not planner.empty:
        pending = (planner["Completed"].astype(str).str.lower() != "true").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessments", len(assessments))
    c2.metric("Average score", f"{avg:.1f}%")
    c3.metric("Overall CGPA", f"{cgpa:.2f}/10")
    c4.metric("Pending study tasks", int(pending))

    st.divider()
    st.subheader("A little progress every day adds up.")
    st.write("Use the tabs to record marks, plan study sessions, calculate grades and keep track of upcoming exams.")

    if not exams.empty:
        upcoming = exams.copy()
        upcoming["Exam Date"] = pd.to_datetime(upcoming["Exam Date"], errors="coerce")
        upcoming = upcoming[upcoming["Exam Date"] >= pd.Timestamp(date.today())].sort_values("Exam Date")
        if not upcoming.empty:
            next_exam = upcoming.iloc[0]
            days = (next_exam["Exam Date"].date() - date.today()).days
            st.info(f"📌 Next exam: **{next_exam['Exam']}** — {days} day(s) remaining.")

    if not assessments.empty:
        st.subheader("Recent assessments")
        st.dataframe(assessments.tail(5).iloc[::-1], use_container_width=True, hide_index=True)

# ======================= CGPA / SGPA =======================
with tabs[1]:
    st.subheader("CGPA & SGPA Calculator")
    st.write("Enter each subject once with its credit value and grade point.")
    with st.form("add_grade_form", clear_on_submit=True):
        left, right = st.columns(2)
        with left:
            semester = st.selectbox("Semester", [f"Semester {i}" for i in range(1, 9)])
            subject = st.text_input("Subject name")
        with right:
            credits = st.number_input("Credits", min_value=1, max_value=10, value=3)
            grade_point = st.number_input("Grade point (0–10)", min_value=0.0, max_value=10.0, value=8.0, step=0.5)
        add_grade = st.form_submit_button("＋ Add subject", use_container_width=True)
        if add_grade:
            if not subject.strip():
                st.error("Please enter a subject name.")
            else:
                grades = load_grades()
                new_row = pd.DataFrame([{
                    "Semester": semester, "Subject": subject.strip(),
                    "Credits": credits, "Grade Point": grade_point
                }])
                grades = pd.concat([grades, new_row], ignore_index=True)
                save_csv(grades, GRADES_FILE)
                st.success("Subject saved.")
                st.rerun()

    grades = load_grades()
    st.divider()
    if grades.empty:
        st.info("No subjects added yet. Add your first subject above.")
    else:
        st.subheader("Saved subject records")
        st.dataframe(grades, use_container_width=True, hide_index=True)
        semester_list = grades["Semester"].dropna().astype(str).unique().tolist()
        selected_semester = st.selectbox("Choose a semester to view SGPA", semester_list)
        a, b = st.columns(2)
        a.metric(f"{selected_semester} SGPA", f"{sgpa_for(grades, selected_semester):.2f}/10")
        b.metric("Overall CGPA", f"{cgpa_for(grades):.2f}/10")

        with st.expander("Remove an incorrect subject entry"):
            options = {
                f"Row {i+1}: {row['Semester']} — {row['Subject']}": i
                for i, row in grades.iterrows()
            }
            selected = st.selectbox("Select entry", list(options.keys()), key="delete_grade_choice")
            if st.button("Delete selected subject", key="delete_grade_btn"):
                grades = grades.drop(index=options[selected]).reset_index(drop=True)
                save_csv(grades, GRADES_FILE)
                st.success("Subject entry deleted.")
                st.rerun()

# ======================= ASSESSMENTS =======================
with tabs[2]:
    st.subheader("Assessment Tracker")
    with st.form("assessment_form", clear_on_submit=True):
        a, b = st.columns(2)
        with a:
            assess_date = st.date_input("Date", value=date.today())
            subject = st.text_input("Subject")
            assessment_name = st.text_input("Assessment name")
        with b:
            total = st.number_input("Total marks", min_value=1.0, value=20.0)
            marks = st.number_input("Marks obtained", min_value=0.0, value=0.0)
        submit = st.form_submit_button("Save assessment", use_container_width=True)
        if submit:
            if not subject.strip() or not assessment_name.strip():
                st.error("Enter both subject and assessment name.")
            elif marks > total:
                st.error("Marks obtained cannot be greater than total marks.")
            else:
                df = load_assessments()
                pct = round((marks / total) * 100, 2)
                df = pd.concat([df, pd.DataFrame([{
                    "Date": str(assess_date), "Subject": subject.strip(),
                    "Assessment": assessment_name.strip(), "Marks": marks,
                    "Total Marks": total, "Percentage": pct
                }])], ignore_index=True)
                save_csv(df, ASSESSMENTS_FILE)
                st.success("Assessment saved.")
                st.rerun()

    assessments = load_assessments()
    st.divider()
    if assessments.empty:
        st.info("Your saved assessments will appear here.")
    else:
        st.dataframe(assessments, use_container_width=True, hide_index=True)
        with st.expander("Delete an assessment"):
            choices = {
                f"{i+1}. {row['Subject']} — {row['Assessment']} ({row['Date']})": i
                for i, row in assessments.iterrows()
            }
            choice = st.selectbox("Select assessment", list(choices.keys()))
            if st.button("Delete selected assessment"):
                assessments = assessments.drop(index=choices[choice]).reset_index(drop=True)
                save_csv(assessments, ASSESSMENTS_FILE)
                st.success("Assessment deleted.")
                st.rerun()

# ======================= PROGRESS =======================
with tabs[3]:
    st.subheader("My Academic Progress")
    assessments = load_assessments()
    if assessments.empty:
        st.info("Add assessment records to see your charts.")
    else:
        assessments["Percentage"] = pd.to_numeric(assessments["Percentage"], errors="coerce")
        st.subheader("Assessment scores")
        chart_data = assessments[["Assessment", "Percentage"]].copy()
        chart_data["Assessment"] = chart_data["Assessment"].astype(str)
        st.bar_chart(chart_data.set_index("Assessment"))
        by_subject = assessments.groupby("Subject")["Percentage"].mean().sort_values(ascending=False)
        st.subheader("Average score by subject")
        st.bar_chart(by_subject)
        x, y, z = st.columns(3)
        x.metric("Assessments", len(assessments))
        y.metric("Average", f"{assessments['Percentage'].mean():.1f}%")
        z.metric("Best score", f"{assessments['Percentage'].max():.1f}%")

# ======================= AI PREDICTOR =======================
with tabs[4]:
    st.subheader("AI Performance Predictor")
    st.write("Enter your academic indicators to receive an estimated performance score.")
    st.caption("This demo model is trained on a small illustrative dataset. Its prediction is for learning/demo purposes, not a reliable forecast.")
    with st.form("predict_form"):
        a, b = st.columns(2)
        with a:
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            study_hours = st.slider("Daily study hours", 0, 12, 3)
            assignment_marks = st.slider("Assignment marks (%)", 0, 100, 75)
        with b:
            internal_marks = st.slider("Internal marks (%)", 0, 100, 70)
            previous_cgpa = st.slider("Previous CGPA", 0.0, 10.0, 7.0, step=0.1)
        predict = st.form_submit_button("Predict performance", use_container_width=True)
    if predict:
        try:
            model = ensure_model()
            row = pd.DataFrame([{
                "attendance": attendance, "study_hours": study_hours,
                "assignment_marks": assignment_marks, "internal_marks": internal_marks,
                "previous_cgpa": previous_cgpa
            }])
            result = float(model.predict(row)[0])
            result = max(0.0, min(100.0, result))
            st.metric("Estimated performance", f"{result:.1f}%")
            st.progress(int(result))
            if result >= 75:
                st.success("Current indicators suggest a stronger performance range.")
            elif result >= 50:
                st.info("Current indicators suggest a middle performance range.")
            else:
                st.warning("Your current indicators suggest there may be room to strengthen your study routine.")
            tips = []
            if attendance < 75: tips.append("Try to improve attendance consistently.")
            if study_hours < 2: tips.append("Build a manageable daily study habit.")
            if assignment_marks < 60: tips.append("Review assignment feedback and practise similar questions.")
            if internal_marks < 60: tips.append("Revise internal-test topics in smaller sessions.")
            if previous_cgpa < 6: tips.append("Set small weekly academic goals and track them.")
            if not tips: tips.append("Keep your routine steady and continue reviewing your progress.")
            st.subheader("Suggestions")
            for tip in tips:
                st.write("• " + tip)
        except Exception as e:
            st.error(f"Could not run the prediction: {e}")

# ======================= WEEKLY STUDY PLANNER =======================
with tabs[5]:
    st.subheader("Weekly Study Planner")
    st.write("Add study sessions and tick them off as you complete them.")
    with st.form("planner_form", clear_on_submit=True):
        a, b = st.columns(2)
        with a:
            task_date = st.date_input("Study date", value=date.today(), key="planner_date")
            task_time = st.time_input("Start time", value=datetime.strptime("17:00", "%H:%M").time())
            task_subject = st.text_input("Subject", key="planner_subject")
        with b:
            task = st.text_input("What will you work on?")
            duration = st.number_input("Duration (minutes)", min_value=5, max_value=600, value=60, step=5)
        add_task = st.form_submit_button("Add study session", use_container_width=True)
        if add_task:
            if not task_subject.strip() or not task.strip():
                st.error("Please enter a subject and a task.")
            else:
                df = load_planner()
                df = pd.concat([df, pd.DataFrame([{
                    "Date": str(task_date), "Time": task_time.strftime("%H:%M"),
                    "Subject": task_subject.strip(), "Task": task.strip(),
                    "Duration": duration, "Completed": False
                }])], ignore_index=True)
                save_csv(df, PLANNER_FILE)
                st.success("Study session added.")
                st.rerun()

    planner = load_planner()
    st.divider()
    if planner.empty:
        st.info("Your weekly study sessions will appear here.")
    else:
        planner["Date"] = pd.to_datetime(planner["Date"], errors="coerce")
        start_week = pd.Timestamp(date.today() - timedelta(days=date.today().weekday()))
        end_week = start_week + pd.Timedelta(days=6)
        this_week = planner[(planner["Date"] >= start_week) & (planner["Date"] <= end_week)].copy()
        this_week = this_week.sort_values(["Date", "Time"])
        st.subheader("This week's sessions")
        if this_week.empty:
            st.info("No sessions planned for this week yet.")
        else:
            for idx, row in this_week.iterrows():
                done = str(row["Completed"]).lower() == "true"
                label = f"{row['Date'].strftime('%a, %d %b')} · {row['Time']} · {row['Subject']} — {row['Task']} ({row['Duration']} min)"
                checked = st.checkbox(label, value=done, key=f"task_{idx}")
                if checked != done:
                    planner_all = load_planner()
                    planner_all.loc[idx, "Completed"] = checked
                    save_csv(planner_all, PLANNER_FILE)
                    st.rerun()
        with st.expander("Delete a study session"):
            all_planner = load_planner()
            task_choices = {
                f"{i+1}. {row['Date']} — {row['Subject']}: {row['Task']}": i
                for i, row in all_planner.iterrows()
            }
            selected_task = st.selectbox("Select session", list(task_choices.keys()))
            if st.button("Delete selected session"):
                all_planner = all_planner.drop(index=task_choices[selected_task]).reset_index(drop=True)
                save_csv(all_planner, PLANNER_FILE)
                st.success("Session deleted.")
                st.rerun()

# ======================= EXAM COUNTDOWN =======================
with tabs[6]:
    st.subheader("Exam Countdown")
    st.write("Add upcoming exams and see how many days remain.")
    with st.form("exam_form", clear_on_submit=True):
        a, b = st.columns(2)
        with a:
            exam_name = st.text_input("Exam name")
            exam_subject = st.text_input("Subject")
        with b:
            exam_date = st.date_input("Exam date", value=date.today() + timedelta(days=7))
            exam_notes = st.text_input("Notes (optional)")
        add_exam = st.form_submit_button("Add exam", use_container_width=True)
        if add_exam:
            if not exam_name.strip() or not exam_subject.strip():
                st.error("Please enter the exam name and subject.")
            else:
                df = load_exams()
                df = pd.concat([df, pd.DataFrame([{
                    "Exam": exam_name.strip(), "Subject": exam_subject.strip(),
                    "Exam Date": str(exam_date), "Notes": exam_notes.strip()
                }])], ignore_index=True)
                save_csv(df, EXAMS_FILE)
                st.success("Exam added.")
                st.rerun()

    exams = load_exams()
    st.divider()
    if exams.empty:
        st.info("No exams added yet.")
    else:
        exams["Exam Date Parsed"] = pd.to_datetime(exams["Exam Date"], errors="coerce")
        exams = exams.sort_values("Exam Date Parsed")
        for _, row in exams.iterrows():
            if pd.isna(row["Exam Date Parsed"]):
                continue
            days_left = (row["Exam Date Parsed"].date() - date.today()).days
            if days_left > 0:
                status = f"{days_left} day(s) remaining"
            elif days_left == 0:
                status = "Today!"
            else:
                status = f"{abs(days_left)} day(s) ago"
            with st.container(border=True):
                st.markdown(f"**{row['Exam']}** · {row['Subject']}")
                st.write(f"📅 {row['Exam Date Parsed'].strftime('%d %B %Y')}  |  ⏳ {status}")
                if str(row["Notes"]).strip() and str(row["Notes"]) != "nan":
                    st.caption(row["Notes"])
        with st.expander("Delete an exam"):
            original = load_exams()
            choices = {
                f"{i+1}. {row['Exam']} — {row['Subject']} ({row['Exam Date']})": i
                for i, row in original.iterrows()
            }
            choice = st.selectbox("Select exam", list(choices.keys()), key="delete_exam_choice")
            if st.button("Delete selected exam", key="delete_exam_btn"):
                original = original.drop(index=choices[choice]).reset_index(drop=True)
                save_csv(original, EXAMS_FILE)
                st.success("Exam deleted.")
                st.rerun()

st.divider()
st.markdown(
    "<div style='text-align:center;color:#8B7A9E;padding:12px;'>"
    "My Academic Space 🎓 · Made with Python and Streamlit"
    "</div>",
    unsafe_allow_html=True
)
