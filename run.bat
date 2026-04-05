@echo off
echo Starting PDF Analyzer...
call .venv\Scripts\activate
streamlit run app.py
pause
