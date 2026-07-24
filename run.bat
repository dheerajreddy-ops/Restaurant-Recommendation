@echo off
title AI Restaurant Recommender
color 0A
echo.
echo  ============================================
echo   AI Restaurant Recommendation System
echo   Deployed Model + Streamlit UI
echo  ============================================
echo.

cd /d "%~dp0"

echo [1] Training model from Zomato dataset...
python train_model.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [2] Starting Streamlit app...
echo     Open http://localhost:8501 in your browser
echo.
streamlit run app.py

pause
