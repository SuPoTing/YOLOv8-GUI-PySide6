# Ultralytics-Pyside6-GUI `V2.5` a GUI for Ultralytics 8.4.70
---
  <p align="center"> 
  English &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-tw.md">繁體中文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-cn.md">简体中文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_jp.md">日文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_kr.md">한국어</a>
 </p>

![](./img/preview.png)

## Environment Setup
### 1. Create a Virtual Environment

Click `create_env.bat` to build a virtual environment with Python 3.10, then activate the environment.

### 2. Run the Application

Click `main.bat` to launch the application.

## Bundling with PyInstaller (via Auto-Py-To-Exe)
### 1. Create a Virtual Environment

Click `create_env.bat` to generate a Python 3.10 virtual environment, then click `activate.bat` to activate it.

### 2. Launch the auto-py-to-exe GUI Interface

```shell
auto-py-to-exe
```

### 3. Add Script Path and Additional Files

Script Location:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\main_en.py
```

Additional Files / Folders:
Click "Add Folder" and select:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\venv\Lib\site-packages\ultralytics
```

Click "Convert .py to .exe".

### 4. Copy Essential Directories
Copy the `config`, `img`, `models`, `ui`, and `utils` folders into:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\output\main
```

### 5. Run main.exe
Execute `main.exe` inside the output directory to launch the compiled application.

## Bundling with Nuitka
### 1. Create a Virtual Environment

Click `create_env-nuitka.bat` to generate a Python 3.10 virtual environment, then click `activate.bat` to activate it.

### 2. Execute the Nuitka Compilation Command

```shell
nuitka --standalone --msvc=latest --lto=yes --enable-plugin=pyside6 --module-parameter=torch-disable-jit=no --include-package=ultralytics main_en.py
```
### 3. Copy Assets Folder

Copy the `.\venv\Lib\site-packages\ultralytics` folder into the newly generated `.dist` directory.

### 4. Run main.exe

Navigate into the `.dist` directory and execute `main.exe` to run the application.

## Important Notes
- `ultralytics` is licensed under the `AGPL-3.0` terms. If you intend to use it for commercial purposes, an official license must be acquired from Ultralytics.
- To deploy your custom weights, you must first train a compatible model architecture via `ultralytics` (supports YOLOv8, YOLOv9 Det/Seg, YOLOv10 Det-only, YOLOv11, YOLOv12, and YOLOv26). Once trained, drop your exported `.pt` weight files into the corresponding subdirectories inside the `models/*` path.
- The software might contain minor bugs. I will continue optimizing codebases and adding interesting features as time permits.
- Exported inference results will automatically be stored under the `./run` path directory.
- The core UI layout asset file is `home.ui`. If you modify the interface using Qt Designer, regenerate the Python source binding with the following command:
```shell
pyside6-uic home.ui > ui/home.py
```
- The application graphic dictionary is `resources.qrc`. If you alter default app icons, recompile the asset file using the following command:
```shell
pyside6-rcc resources.qrc > ui/resources_rc.py
```
- Oriented Bounding Boxes (OBB) functionality operates under `Detect mode`. To load your custom OBB models, the filename must explicitly contain the substring `obb` (e.g., `yolov8n-obb.pt`). Files without `obb` in their names will fall back to standard horizontal detection.

## Implemented Features
### 1.Pipeline Task Selection
- Image Classification (Classify)
- Object Detection (Detect)
- Oriented Object Detection (OBB)
- Pose Estimation (Pose)
- Instance Segmentation (Segment)
- Object Tracking (Track)
### 2.Stream & Context Data Feeds
- Single file media evaluation channel (Images / Videos)
- Directory batch parsing and evaluation pipeline
- Native UI drag-and-drop ingestion file handler
- Integrated hardware peripherals support (Webcams / USB Cameras)
- Built-in live stream bindings using `chose_rtsp` and `load_rtsp` function hooks

## Future Roadmap
- [ ] Implement live tracking for local host hardware and GPU utilization metrics.
- [ ] Add interactive data visualization charts displaying dynamic target counts over timelines.

## References
- [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [PySide6-YOLOv8](https://github.com/Jai-wei/YOLOv8-PySide6-GUI/tree/main)
- [YOLOSHOW](https://github.com/SwimmingLiu/YOLOSHOW/tree/31644373fca58aefcc9dba72a610c92031e5331b)
