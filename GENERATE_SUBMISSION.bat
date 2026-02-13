@echo off
REM Generate Submission CSV - Run this when you have test data!

echo ================================================================================
echo SUBMISSION GENERATION
echo ================================================================================
echo.
echo This will generate a submission.csv file from your test data.
echo Make sure you have:
echo   - test.csv (with 'index' column)
echo   - test_images folder (with test images)
echo.

set /p continue="Continue? (y/n): "
if /i not "%continue%"=="y" exit /b

REM Activate virtual environment
echo.
echo Activating environment...
call chipsal_env\Scripts\activate.bat

REM Run submission generation
echo Generating submission...
cd abiral
python quick_test.py --mode submission --csv ..\test.csv --images ..\test_images --output ..\submission.csv

echo.
echo ================================================================================
echo Done! Check submission.csv in the main folder.
echo ================================================================================
pause
