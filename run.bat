@echo off

python.exe -m venv venv

call .\venv\Scripts\activate.bat

python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt

if %errorlevel% equ 0 (
   python.exe app.py --xformers --setup_mode --share%*
)

if %errorlevel% neq 0 (
    echo.
    echo.
    echo.
    pause
)