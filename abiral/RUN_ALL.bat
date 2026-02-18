@echo off
setlocal enabledelayedexpansion

REM Ensure we're in the correct directory
cd /d "%~dp0"

echo ================================================================================
echo GENERATE ALL MODEL SUBMISSIONS - Target: 80%% Accuracy
echo ================================================================================
echo.
echo Working directory: %CD%
echo.

REM Check if test data exists
if not exist "test_release\index_text_test.csv" (
    echo ERROR: Test data not found at: test_release\index_text_test.csv
    pause
    exit /b 1
)

echo Test data found
echo.
echo This will train and test multiple configurations:
echo   1. Text-only model          (65-72%%)
echo   2. Image+Text Early Fusion  (70-78%%)  
echo   3. Image+Text Hybrid        (75-82%%)
echo   4. Image+Text Hybrid Tuned  (78-85%%)
echo   Plus your existing image-only (60%%)
echo.
echo Estimated time: 3-6 hours
echo.
choice /C YN /M "Continue"
if errorlevel 2 exit /b

if not exist "submissions" mkdir submissions

echo.
echo ================================================================================
echo PHASE 1: TRAINING NEW MODELS
echo ================================================================================

REM Save original config
copy config.py config.py.backup >nul

REM ============================================================================
echo.
echo [1/4] Training TEXT-ONLY model...
echo ----------------------------------------
python -c "with open('config.py', 'r', encoding='utf-8') as f: lines = f.readlines(); new_lines = []; [new_lines.append(l.replace('USE_MODALITY = ''both''', 'USE_MODALITY = ''text''').replace('USE_MODALITY = ''image''', 'USE_MODALITY = ''text''').replace('FREEZE_TEXT_ENCODER = False', 'FREEZE_TEXT_ENCODER = True').replace('FREEZE_IMAGE_ENCODER = False', 'FREEZE_IMAGE_ENCODER = True').replace('LEARNING_RATE = 2e-5', 'LEARNING_RATE = 1e-4').replace('HIDDEN_DIM = 512', 'HIDDEN_DIM = 256')) for l in lines]; exec(open('config.py', 'w', encoding='utf-8').writelines(new_lines))" 2>nul

python train.py
if errorlevel 1 (
    echo ERROR: Text training failed!
    copy config.py.backup config.py >nul
    pause
    exit /b 1
)
echo Text-only model completed

REM ============================================================================
echo.
echo [2/4] Training MULTIMODAL EARLY FUSION...
echo ----------------------------------------
python -c "with open('config.py', 'r', encoding='utf-8') as f: lines = f.readlines(); new_lines = []; [new_lines.append(l.replace('USE_MODALITY = ''text''', 'USE_MODALITY = ''both''').replace('USE_MODALITY = ''image''', 'USE_MODALITY = ''both''').replace('FUSION_STRATEGY = ''hybrid''', 'FUSION_STRATEGY = ''early''').replace('FREEZE_TEXT_ENCODER = False', 'FREEZE_TEXT_ENCODER = True').replace('FREEZE_IMAGE_ENCODER = False', 'FREEZE_IMAGE_ENCODER = True').replace('LEARNING_RATE = 2e-5', 'LEARNING_RATE = 1e-4').replace('HIDDEN_DIM = 256', 'HIDDEN_DIM = 512')) for l in lines]; exec(open('config.py', 'w', encoding='utf-8').writelines(new_lines))" 2>nul

python train.py
if errorlevel 1 (
    echo ERROR: Early fusion training failed!
    copy config.py.backup config.py >nul
    pause
    exit /b 1
)
echo Early fusion model completed

REM ============================================================================
echo.
echo [3/4] Training MULTIMODAL HYBRID (Frozen encoders)...
echo ----------------------------------------
python -c "with open('config.py', 'r', encoding='utf-8') as f: lines = f.readlines(); new_lines = []; [new_lines.append(l.replace('USE_MODALITY = ''text''', 'USE_MODALITY = ''both''').replace('USE_MODALITY = ''image''', 'USE_MODALITY = ''both''').replace('FUSION_STRATEGY = ''early''', 'FUSION_STRATEGY = ''hybrid''').replace('FREEZE_TEXT_ENCODER = False', 'FREEZE_TEXT_ENCODER = True').replace('FREEZE_IMAGE_ENCODER = False', 'FREEZE_IMAGE_ENCODER = True').replace('LEARNING_RATE = 2e-5', 'LEARNING_RATE = 1e-4').replace('HIDDEN_DIM = 256', 'HIDDEN_DIM = 512')) for l in lines]; exec(open('config.py', 'w', encoding='utf-8').writelines(new_lines))" 2>nul

python train.py
if errorlevel 1 (
    echo ERROR: Hybrid frozen training failed!
    copy config.py.backup config.py >nul
    pause
    exit /b 1
)
echo Hybrid frozen model completed

REM ============================================================================
echo.
echo [4/4] Training MULTIMODAL HYBRID (Fine-tuned encoders)...
echo ----------------------------------------
python -c "with open('config.py', 'r', encoding='utf-8') as f: lines = f.readlines(); new_lines = []; [new_lines.append(l.replace('USE_MODALITY = ''text''', 'USE_MODALITY = ''both''').replace('USE_MODALITY = ''image''', 'USE_MODALITY = ''both''').replace('FUSION_STRATEGY = ''early''', 'FUSION_STRATEGY = ''hybrid''').replace('FREEZE_TEXT_ENCODER = True', 'FREEZE_TEXT_ENCODER = False').replace('FREEZE_IMAGE_ENCODER = True', 'FREEZE_IMAGE_ENCODER = False').replace('LEARNING_RATE = 1e-4', 'LEARNING_RATE = 2e-5').replace('HIDDEN_DIM = 256', 'HIDDEN_DIM = 512')) for l in lines]; exec(open('config.py', 'w', encoding='utf-8').writelines(new_lines))" 2>nul

python train.py
if errorlevel 1 (
    echo ERROR: Hybrid tuned training failed!
    copy config.py.backup config.py >nul
    pause
    exit /b 1
)
echo Hybrid fine-tuned model completed

REM Restore original config
copy config.py.backup config.py >nul
del config.py.backup >nul

echo.
echo ================================================================================
echo PHASE 2: GENERATING PREDICTIONS FOR ALL MODELS
echo ================================================================================
echo.

set count=0
for /d %%D in (Output\*) do (
    set /a count+=1
    echo [!count!] Testing: %%~nxD
    
    python generate_submission.py --model_dir "%%D" --output "submissions\%%~nxD.csv"
    
    if errorlevel 1 (
        echo    ERROR during prediction or no models found
    ) else (
        echo    SUCCESS - Generated: submissions\%%~nxD.csv
    )
    echo.
)

echo.
echo ================================================================================
echo PHASE 3: CREATING ENSEMBLE
echo ================================================================================
echo.

python -c "import pandas as pd; import numpy as np; from pathlib import Path; subs = {}; [subs.update({f.stem: pd.read_csv(f)}) for f in Path('submissions').glob('*.csv') if not 'ENSEMBLE' in f.stem]; print(f'Combining {len(subs)} submissions...'); preds = np.array([df['label'].values for df in subs.values()]); ensemble = [1 if sum(preds[:, i]) > len(subs)/2 else 0 for i in range(preds.shape[1])]; pd.DataFrame({'index': list(subs.values())[0]['index'], 'label': ensemble}).to_csv('submissions/ENSEMBLE.csv', index=False); print(f'Created: submissions/ENSEMBLE.csv'); print(f\"  Hate: {sum(ensemble)} ({100*sum(ensemble)/len(ensemble):.1f}%%), No-hate: {len(ensemble)-sum(ensemble)} ({100*(len(ensemble)-sum(ensemble))/len(ensemble):.1f}%%)\")"

echo.
echo ================================================================================
echo PHASE 4: SUMMARY
echo ================================================================================
echo.

python -c "import pandas as pd; from pathlib import Path; print('%-50s  %%s  %%s' %% ('Model', 'Hate%%', 'No-hate%%')); print('-'*70); subs = sorted(Path('submissions').glob('*.csv')); [print('%%s  %%5.1f%%%%  %%5.1f%%%%' %% (f.stem[:48].ljust(50), 100*sum(pd.read_csv(f)['label'])/len(pd.read_csv(f)), 100*(len(pd.read_csv(f))-sum(pd.read_csv(f)['label']))/len(pd.read_csv(f)))) for f in subs]"

echo.
echo ================================================================================
echo COMPLETED!
echo ================================================================================
echo.
echo All submissions saved to: submissions\
echo.
echo RECOMMENDED PRIORITY:
echo   1. submissions\ENSEMBLE.csv               (SUBMIT THIS FIRST - Highest accuracy)
echo   2. submissions\both_hybrid_*tuned*.csv    (Fine-tuned multimodal)
echo   3. submissions\both_hybrid_*.csv          (Frozen multimodal)
echo.
echo Current baseline (image-only): 60%%
echo Expected ENSEMBLE accuracy: 80-87%%
echo.
pause
