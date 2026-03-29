@echo off
setlocal enableextensions

set "RUN_ID=%~1"
if "%RUN_ID%"=="" set "RUN_ID=20260320_210439"

set "PYTHON_CMD=%~2"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%" >nul
if errorlevel 1 (
  echo [SAFE_RL] Failed to enter repo root: %ROOT_DIR%
  exit /b 1
)

echo [SAFE_RL] Repo root: %CD%
echo [SAFE_RL] Run ID: %RUN_ID%
echo [SAFE_RL] Python: %PYTHON_CMD%
echo.

call :run_step "Stage 1 - build pointwise and stage1 probe data" "%PYTHON_CMD%" safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id %RUN_ID%
call :run_step "Stage 5 - bootstrap strong pairs" "%PYTHON_CMD%" safe_rl_main.py --config safe_rl/config/stage5_pair_bootstrap.yaml --stage stage5 --run-id %RUN_ID%
call :run_step "Stage 2 - world-focused v2 training" "%PYTHON_CMD%" safe_rl_main.py --config safe_rl/config/stage2_v2_world_pair_focus.yaml --stage stage2 --run-id %RUN_ID%
call :run_step "Stage 5 - held-out after-trace validation" "%PYTHON_CMD%" safe_rl_main.py --config safe_rl/config/shield_trace_holdout_c1.yaml --stage stage5 --run-id %RUN_ID%

echo.
echo [SAFE_RL] All 4 steps completed successfully.
popd >nul
exit /b 0

:run_step
set "STEP_NAME=%~1"
echo ============================================================
echo [SAFE_RL] %STEP_NAME%
echo ============================================================
shift
call %*
if errorlevel 1 (
  echo.
  echo [SAFE_RL] Step failed: %STEP_NAME%
  popd >nul
  exit /b 1
)
echo.
exit /b 0
