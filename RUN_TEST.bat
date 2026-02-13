@echo off
REM Quick Test Script - Run this to test your trained model!

echo ================================================================================
echo QUICK TEST - Testing your trained MoE model
echo ================================================================================
echo.

REM Activate virtual environment
echo Activating environment...
call chipsal_env\Scripts\activate.bat

REM Run quick test
echo Running test on demo image...
cd abiral
python quick_test.py

echo.
echo ================================================================================
echo Test complete! Check the results above.
echo ================================================================================
echo.
echo To test on submission data (when you have test.csv and test_images):
echo   python quick_test.py --mode submission --csv test.csv --images test_images
echo.
pause
