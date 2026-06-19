# Ultralytics-Pyside6-GUI `V2.5` a GUI for Ultralytics 8.4.70
---
<p align="center"> 
  <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README.md"> English</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-tw.md">繁體中文</a> &nbsp; | &nbsp; 简体中文</a>
 </p>

![](./img/preview.png)

## 实验环境
### 1. 建立虚拟环境

点击`create_env.bat`建立一个python3.10以上的虚拟环境，然后启动环境。

### 2. 执行程序

点击`main.bat`

## 打包
### 1. 建立虚拟环境

点击`create_env.bat`建立一个python3.10以上的虚拟环境，然后点击`activate.bat`启动环境。

### 2. 启动auto-py-to-exe UI界面

```shell
auto-py-to-exe
```

### 3. 添加脚本位置以及附加文件

脚本位置
```shell
(your YOLOv8-GUI-PySide6-main PATH)\main.py
```

附加文件点选新增目录
```shell
(your YOLOv8-GUI-PySide6-main PATH)\venv\Lib\site-packages\ultralytics
```

点选转换

### 5. 复制文件
将`conig`、`img`、`models`、`ui`、`utils`复制到`(your YOLOv8-GUI-PySide6-main PATH)\output\main`

### 6. 启动main.exe
运行main.exe以启动应用程序。

## 打包-Nuitka
### 1. 建立虚拟环境

点击`create_env-nuitka.bat`建立一个python3.10以上的虚拟环境，然后点击`activate.bat`启动环境。

### 2. 输入nuitka指令

```shell
nuitka --standalone --msvc=latest --lto=yes --enable-plugin=pyside6 --module-parameter=torch-disable-jit=no --include-package=ultralytics main.py
```

### 3. 複製檔案

将`.\venv\Lib\site-packages\ultralytics`文件夹复制到`.dist`文件夹

### 4. 启动main.exe

进入`.dist`运行`main.exe`以启动应用程序。

## 注意事項
- `ultralytics`遵循`AGPL-3.0`，如果需要商业用途，需要取得其license。
- 如果您希望使用自己的model，则需要先使用`ultralytics`训练yolov8/9(det&seg)/10(only det)/11/12/26的model，然后将训练好的`.pt`放入`models/*`文件夹中。
- 软件可能存在一些bug，我会在时间允许的情况下继续优化并添加一些更有趣的功能。
- 如果您有储存检测结果，它们将保存在`./run`路径中。
- UI设计文件为`home.ui`，如果对UI重新布局后需转换成py，需要在虚拟环境输入`pyside6-uic home.ui > ui/home.py`指令重新生成`.py`文件。
- 资源文件为`resources.qrc`，如果修改默认icon，需要在虚拟环境输入`pyside6-rcc resources.qrc > ui/resources_rc.py`指令重新產生`.py`文件。
- 旋转框模式在`Detect mode`，如果要使用自己训练的obb模型，需要在文件名中加`obb`，如`yolov8n-obb.pt`，未加`obb`只会进入一般侦测模式。

## 现有功能
### 1.模式选择
- 图像分类
- 物体检测
- 物体检测(OBB)
- 姿态检测
- 实例分割
- 目标追踪
### 2.数据输入方式
- 单一文件检测功能
- 文件夹(批处理)检测功能
- 支持拖拽文件输入
- 输入支持Camera
- 支持`chose_rtsp`、`load_rtsp`函数

## 未来方向
- [ ] 监控系统硬件使用情况
- [ ] 显示目标数量变化的图表

## 参考文献
- [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [PySide6-YOLOv8](https://github.com/Jai-wei/YOLOv8-PySide6-GUI/tree/main)
- [YOLOSHOW](https://github.com/SwimmingLiu/YOLOSHOW/tree/31644373fca58aefcc9dba72a610c92031e5331b)

