@echo off
REM Quick model selection helper

echo ================================================================================
echo MODEL SELECTION HELPER
echo ================================================================================
echo.
echo What would you like to do?
echo.
echo   1. List all available models
echo   2. Interactive selection (guided)
echo   3. Test specific model folder
echo   4. Test specific fold
echo   5. Exit
echo.

choice /C 12345 /N /M "Select option [1-5]: "

if errorlevel 5 exit /b
if errorlevel 4 goto TEST_FOLD
if errorlevel 3 goto TEST_MODEL
if errorlevel 2 goto INTERACTIVE
if errorlevel 1 goto LIST

:LIST
echo.
python generate_submission.py --list-models
pause
exit /b

:INTERACTIVE
echo.
python generate_submission.py --interactive
pause
exit /b

:TEST_MODEL
echo.
echo Enter model folder path (e.g., Output/image_original_20260212_132844)
set /p MODEL_DIR="Model folder: "
echo.
echo Enter output filename (e.g., my_test.csv)
set /p OUTPUT_FILE="Output file: "
echo.
python generate_submission.py --model_dir "%MODEL_DIR%" --output "%OUTPUT_FILE%"
pause
exit /b

:TEST_FOLD
echo.
echo Enter fold number(s) (e.g., 0 or 0,1,2 or all)
set /p FOLD_NUM="Fold(s): "
echo.
echo Enter output filename (e.g., fold0_test.csv)
set /p OUTPUT_FILE="Output file: "
echo.
python generate_submission.py --fold %FOLD_NUM% --output "%OUTPUT_FILE%"
pause
exit /b
