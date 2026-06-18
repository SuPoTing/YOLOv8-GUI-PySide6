@echo off

REM 創建 Python 原生虛擬環境
python -m venv venv

REM 獲得當前路徑並激活虛擬環境
call venv\Scripts\activate.bat

REM 安裝依賴套件
python.exe -m pip install --upgrade pip
pip install pyside6
pip install chardet
pip install pytube
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics==8.4.70
pip install lapx
pip install Nuitka

REM 提示安裝完成
echo Complete

REM 保持命令提示符窗口開啟
pause
