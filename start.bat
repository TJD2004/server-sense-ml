@echo off
echo ===============================================
echo   SolarSense ML Microservice (Port 8000)
echo ===============================================
set PYTHONIOENCODING=utf-8
"C:\Users\Tushar.LAPTOP-MCPUGNCD\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
