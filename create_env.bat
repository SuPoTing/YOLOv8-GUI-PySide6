@echo off

REM 創建 Python 原生虛擬環境
python -m venv venv

REM 獲得當前路徑並激活虛擬環境
call venv\Scripts\activate.bat

REM 安裝依賴套件
pip install pyside6
pip install chardet
pip install pytube
pip install ultralytics
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install moviepy==1.0.3
pip install numpy==1.26.4
pip install lapx
pip install auto-py-to-exe

REM 提示安裝完成
echo Complete

REM 保持命令提示符窗口開啟
pause
