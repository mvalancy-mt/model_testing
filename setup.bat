@echo off
echo Installing Python dependencies from requirements.txt...

pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please check if pip is installed correctly.
    exit /b 1
)

echo Dependencies installed successfully.
