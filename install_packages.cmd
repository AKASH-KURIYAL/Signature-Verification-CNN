@echo off
echo Installing Python packages from requirements.txt...

:: Check if requirements.txt exists
if not exist requirement.txt (
    echo requirement.txt not found.
    exit /b 1
)

:: Install the packages
pip install -r requirement.txt

:: Check if the installation was successful
if %errorlevel% neq 0 (
    echo Failed to install packages.
    exit /b 1
)

echo All packages installed successfully.
pause
