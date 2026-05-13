@echo off
cd /d "%~dp0"
echo ========================================
echo  PhoneScope DB Diagnostics
echo ========================================
echo.
echo [1] Port 3306 (system MySQL):
netstat -ano | findstr ":3306 "
echo.
echo [2] Port 3307 (Docker MySQL):
netstat -ano | findstr ":3307 "
echo.
echo [3] Docker MySQL container status:
docker ps -a --filter "name=phonescope-mysql" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
echo [4] Starting MySQL container (port 3307)...
docker compose -p phonescope up -d
echo.
echo [5] Waiting 15s for MySQL to initialize...
timeout /t 15 /nobreak
echo.
echo [6] Testing DB connection on port 3307:
venv\Scripts\python.exe -c "import pymysql; c=pymysql.connect(host='127.0.0.1',port=3307,user='phonescope',password='phonescope',database='phonescope'); print('MySQL connection: OK'); c.close()"
echo.
pause
