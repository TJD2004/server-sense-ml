$env:PYTHONIOENCODING = "utf-8"
Write-Host "Starting SolarSense ML Microservice on port 8000..." -ForegroundColor Green
& "C:\Users\Tushar.LAPTOP-MCPUGNCD\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
