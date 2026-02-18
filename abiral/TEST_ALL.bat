@echo off
setlocal enabledelayedexpansion

REM Ensure we're in the correct directory
cd /d "%~dp0"

echo ================================================================================
echo TEST ALL EXISTING MODELS - Generate Submissions
echo ================================================================================
echo.
echo Working directory: %CD%
echo.

REM Check if test data exists
if not exist "test_release\index_text_test.csv" (
    echo ERROR: Test data not found!
    echo Expected: test_release\index_text_test.csv
    echo Please ensure test_release folder exists in: %CD%
    pause
    exit /b 1
)

if not exist "test_release\test_images" (
    echo ERROR: Test images not found!
    echo Expected: test_release\test_images\
    pause
    exit /b 1
)

echo Test data found: test_release\index_text_test.csv
echo Test images found: test_release\test_images\
echo.
echo This will generate predictions from ALL trained models
echo No training - just testing with existing models
echo.
echo Estimated time: 10-20 minutes
echo.
pause

if not exist "submissions" mkdir submissions

echo.
echo ================================================================================
echo TESTING ALL MODELS
echo ================================================================================
echo.

set count=0
for /d %%D in (Output\*) do (
    set /a count+=1
    echo [!count!] Testing: %%~nxD
    echo ----------------------------------------
    
    python generate_submission.py --model_dir "%%D" --output "submissions\%%~nxD.csv"
    
    if errorlevel 1 (
        echo    ERROR during prediction or no models found
    ) else (
        echo    SUCCESS - Generated: submissions\%%~nxD.csv
    )
    echo.
)

if !count! == 0 (
    echo ERROR: No models found in Output/ folder!
    echo Please train models first using RUN_ALL.bat
    pause
    exit /b 1
)

echo ================================================================================
echo CREATING ENSEMBLE
echo ================================================================================
echo.

python -c "import pandas as pd; import numpy as np; from pathlib import Path; subs = {}; [subs.update({f.stem: pd.read_csv(f)}) for f in Path('submissions').glob('*.csv') if not 'ENSEMBLE' in f.stem]; print(f'Combining {len(subs)} submissions...'); preds = np.array([df['label'].values for df in subs.values()]); ensemble = [1 if sum(preds[:, i]) > len(subs)/2 else 0 for i in range(preds.shape[1])]; pd.DataFrame({'index': list(subs.values())[0]['index'], 'label': ensemble}).to_csv('submissions/ENSEMBLE.csv', index=False); print(f'Created: submissions/ENSEMBLE.csv'); print(f\"  Hate: {sum(ensemble)} ({100*sum(ensemble)/len(ensemble):.1f}%%), No-hate: {len(ensemble)-sum(ensemble)} ({100*(len(ensemble)-sum(ensemble))/len(ensemble):.1f}%%)\")"

echo.
echo ================================================================================
echo RESULTS SUMMARY
echo ================================================================================
echo.

python -c "import pandas as pd; from pathlib import Path; print('%-50s  %%s  %%s' %% ('Submission File', 'Hate%%', 'No-hate%%')); print('-'*70); subs = sorted(Path('submissions').glob('*.csv')); [print('%%s  %%5.1f%%%%  %%5.1f%%%%' %% (f.stem[:48].ljust(50), 100*sum(pd.read_csv(f)['label'])/len(pd.read_csv(f)), 100*(len(pd.read_csv(f))-sum(pd.read_csv(f)['label']))/len(pd.read_csv(f)))) for f in subs]"

echo.
echo ================================================================================
echo COMPLETED!
echo ================================================================================
echo.
echo All submissions saved to: submissions\
echo.
echo Files generated: !count! model(s) + 1 ensemble
echo.
echo RECOMMENDED TO SUBMIT:
echo   1. submissions\ENSEMBLE.csv  (Best - combines all models)
echo.
dir /b submissions\*.csv
echo.
pause
