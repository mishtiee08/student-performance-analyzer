MY ACADEMIC SPACE — FRESH PROJECT

1. Extract this ZIP into a NEW folder.
2. Open that extracted folder in VS Code (File > Open Folder).
3. Open Terminal > New Terminal.
4. Create/activate a virtual environment if desired:
   python -m venv venv
   venv\Scripts\activate
5. Install packages:
   pip install -r requirements.txt
6. Run:
   python -m streamlit run app.py

The app creates its own data/ and models/ files automatically.
It does not need the old modules/ folder or your previous app.py.

Features:
- Dashboard
- CGPA and SGPA calculator
- Assessment tracker
- Progress charts
- Demo machine-learning performance predictor
- Weekly study planner
- Exam countdown

Important:
- This is a local, single-user student project. Do not deploy it publicly
  with private student information without adding authentication and
  proper per-user data storage.
- The predictor uses a small illustrative sample dataset. It is a demo,
  not a validated prediction tool.
- Keep the data/ folder if you want to retain your saved records.
