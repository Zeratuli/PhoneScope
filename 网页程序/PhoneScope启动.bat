@echo off
chcp 65001 >nul 2>nul
title PhoneScope Launcher
cd /d "%~dp0"
set "BACKEND_PID_FILE=%TEMP%\phonescope_backend.pid"
set "FRONTEND_PID_FILE=%TEMP%\phonescope_frontend.pid"

echo ========================================================
echo    PhoneScope - Smart Phone Recognition System
echo ========================================================
echo.

echo [1/6] Cleaning up previous sessions...
if exist "%BACKEND_PID_FILE%" (
    for /f %%a in ('type "%BACKEND_PID_FILE%"') do taskkill /PID %%a /T /F >nul 2>nul
    del /f /q "%BACKEND_PID_FILE%" >nul 2>nul
)
if exist "%FRONTEND_PID_FILE%" (
    for /f %%a in ('type "%FRONTEND_PID_FILE%"') do taskkill /PID %%a /T /F >nul 2>nul
    del /f /q "%FRONTEND_PID_FILE%" >nul 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /T /F >nul 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /T /F >nul 2>nul
)
taskkill /FI "WINDOWTITLE eq PhoneScope Backend*" /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq PhoneScope Frontend*" /F >nul 2>nul
echo   [OK] Cleanup done

echo [2/6] Checking Docker...
docker info >nul 2>nul
if not %errorlevel%==0 (
    echo   [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)
echo   [OK] Docker is ready

echo [3/6] Starting MySQL on port 3307...
docker compose -p phonescope up -d
if not %errorlevel%==0 (
    echo   [ERROR] Failed to start MySQL container.
    pause
    exit /b 1
)
echo   [OK] MySQL starting...
timeout /t 3 /nobreak >nul

echo [4/6] Starting FastAPI backend...
set "BACKEND_DIR=%~dp0backend"
powershell -NoProfile -Command "$p = Start-Process cmd.exe -WorkingDirectory '%BACKEND_DIR%' -ArgumentList '/k','title PhoneScope Backend && chcp 65001 >nul && set PYTHONIOENCODING=utf-8 && set PYTHONUNBUFFERED=1 && venv\\Scripts\\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 || (echo [ERROR] Backend failed to start & pause)' -PassThru; Set-Content -Path '%BACKEND_PID_FILE%' -Value $p.Id" >nul
echo   [OK] Backend starting on http://localhost:8000

echo [5/6] Starting Vite frontend...
set "FRONTEND_DIR=%~dp0frontend"
powershell -NoProfile -Command "$p = Start-Process cmd.exe -WorkingDirectory '%FRONTEND_DIR%' -ArgumentList '/k','title PhoneScope Frontend && npm run dev || (echo [ERROR] Frontend failed to start & pause)' -PassThru; Set-Content -Path '%FRONTEND_PID_FILE%' -Value $p.Id" >nul
echo   [OK] Frontend starting on http://localhost:5173

echo [6/6] Waiting for services to be ready...
set cnt=0
:wait_loop
if %cnt% geq 30 goto wait_timeout
timeout /t 2 /nobreak >nul
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8000/api/v1/health' -TimeoutSec 3 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
if %errorlevel%==0 goto wait_ok
set /a cnt+=1
set /p="." <nul
goto wait_loop
:wait_ok
echo.
echo   [OK] Backend is ready!
goto open_browser
:wait_timeout
echo.
echo   [WARN] Backend health check timed out, opening browser anyway...

:open_browser
timeout /t 2 /nobreak >nul
start http://localhost:5173

echo.
echo ========================================================
echo   PhoneScope is running!
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   MySQL:     localhost:3307
echo.
echo   Press any key to STOP all services.
echo ========================================================
echo.
pause

echo Stopping services...
if exist "%BACKEND_PID_FILE%" (
    for /f %%a in ('type "%BACKEND_PID_FILE%"') do taskkill /PID %%a /T /F >nul 2>nul
    del /f /q "%BACKEND_PID_FILE%" >nul 2>nul
)
if exist "%FRONTEND_PID_FILE%" (
    for /f %%a in ('type "%FRONTEND_PID_FILE%"') do taskkill /PID %%a /T /F >nul 2>nul
    del /f /q "%FRONTEND_PID_FILE%" >nul 2>nul
)
taskkill /FI "WINDOWTITLE eq PhoneScope Backend*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq PhoneScope Frontend*" /T /F >nul 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /T /F >nul 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /T /F >nul 2>nul
)
docker compose -p phonescope down
echo All services stopped.
timeout /t 3
