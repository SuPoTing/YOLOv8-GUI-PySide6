@echo off
REM 激活 YOLOv8 環境
call venv\Scripts\activate.bat

REM 運行 main.py 腳本
python main.py

REM 保持窗口打開以顯示任何錯誤信息
pause
