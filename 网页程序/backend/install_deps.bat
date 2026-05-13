@echo off
cd /d "%~dp0"
echo Installing all dependencies into venv...
venv\Scripts\python.exe -m pip install sqlalchemy pymysql cryptography opencv-python-headless timm
echo.
echo Testing imports...
venv\Scripts\python.exe -c "from app.main import app; print('OK: backend imports successful')"
echo.
echo Verifying model paths...
venv\Scripts\python.exe -c "from app.config import YOLO_MODEL_PATH, CLASSIFIER_MODEL_PATH, CLASSIFIER_CLASSES_PATH, CLASSIFIER_MODEL_NAME; import os; print('YOLO:', 'OK' if os.path.exists(YOLO_MODEL_PATH) else 'MISSING - '+YOLO_MODEL_PATH); print('Classifier:', 'OK' if os.path.exists(CLASSIFIER_MODEL_PATH) else 'MISSING - '+CLASSIFIER_MODEL_PATH); print('Classes:', 'OK' if os.path.exists(CLASSIFIER_CLASSES_PATH) else 'MISSING - '+CLASSIFIER_CLASSES_PATH); print('Model name:', CLASSIFIER_MODEL_NAME)"
echo.
pause
