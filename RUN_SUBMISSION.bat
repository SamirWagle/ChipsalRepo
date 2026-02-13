@echo off
REM Generate Submission CSV - Simple Script

echo ================================================================================
echo GENERATING SUBMISSION CSV
echo ================================================================================
echo.
echo This will create submission.csv from your test_release data
echo Using your trained models in abiral/Output/
echo.

REM Activate virtual environment
call chipsal_env\Scripts\activate.bat

REM Go to abiral directory
cd abiral

REM Run submission generation
echo Running prediction...
echo.
python generate_submission.py

echo.
echo ================================================================================
echo Done! Check submission.csv in the abiral folder
echo ================================================================================
pause
