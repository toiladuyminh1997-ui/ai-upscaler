@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem First run: create a private Python environment and install everything into it.
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>&1 || (
        echo Chua co Python. Tai tai https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), roi chay lai file nay.
        start https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo [1/3] Tao moi truong Python ...
    py -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    echo [2/3] Cai PyTorch ^(lan dau mat vai phut, ~2.5 GB^) ...
    where nvidia-smi >nul 2>&1 && (
        ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto :fail
    ) || (
        ".venv\Scripts\python.exe" -m pip install torch || goto :fail
    )
    echo [3/3] Cai thu vien con lai ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

echo.
echo  *** KHONG DONG CUA SO NAY khi dang dung app - dong lai la app ngung chay. ***
echo.
".venv\Scripts\python.exe" app.py
echo Server da dung. Neu co loi, chup man hinh cua so nay.
pause
exit /b 0

:fail
echo Cai dat that bai. Xoa thu muc .venv roi chay lai run.bat.
rmdir /s /q .venv 2>nul
pause
exit /b 1
