
# Ultralytics-Pyside6-GUI `V2.5` a GUI for Ultralytics 8.4.70
---
  <p align="center"> 
  <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README.md">English</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-tw.md">繁體中文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-cn.md">简体中文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_jp.md">日文</a>  &nbsp; | &nbsp; 한국어</a>
 </p>

![](./img/preview.png)

## 환경 설정
### 1. 가상 환경 생성
`create_env.bat`를 클릭하여Python 3.10기반의 가상 환경을 구축하고 환경을 활성화합니다.

### 2. 애플리케이션 실행
`main.bat`를 클릭하여 애플리케이션을 실행합니다.

## PyInstaller를 이용한 패키징 (Auto-Py-To-Exe 경유)
### 1. 가상 환경 생성
`create_env.bat`를 클릭하여 Python 3.10 가상 환경을 생성한 후, `activate.bat`를 클릭하여 활성화합니다.

### 2. auto-py-to-exe GUI 인터페이스 실행
```shell
auto-py-to-exe
```

### 3. 스크립트 경로 및 추가 파일 추가
Script Location:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\main.py
```

Additional Files / Folders:
'Add Folder'를 클릭하고 다음을 선택합니다:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\venv\Lib\site-packages\ultralytics
```

'Convert .py to .exe'를 클릭합니다.

### 4. 필수 디렉터리 복사
`config`, `img`, `models`, `ui`, `utils`폴더를 다음 경로로 복사합니다:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\output\main
```

### 5. main.exe 실행
출력 디렉터리 내의`main.exe`를 실행하여 컴파일된 애플리케이션을 구동합니다.

## Nuitka를 이용한 패키징
### 1. 가상 환경 생성
`create_env-nuitka.bat`를 클릭하여Python 3.10가상 환경을 생성한 후, `activate.bat`를 클릭하여 활성화합니다.

## Nuitka 컴파일 명령 실행
```shell
nuitka --standalone --msvc=latest --lto=yes --enable-plugin=pyside6 --module-parameter=torch-disable-jit=no --include-package=ultralytics main.py
```
### 2. 에셋 폴더 복사
`.\venv\Lib\site-packages\ultralytics`폴더를 새로 생성된`.dist`디렉터리 내부로 복사합니다.

### 4. main.exe 실행
`.dist`디렉터리로 이동한 후`main.exe`를 실행하여 애플리케이션을 구동합니다.

## 중요 참고 사항
- `ultralytics`는`AGPL-3.0`라이선스 조건에 따라 라이선스가 부여됩니다. 상업적 목적으로 사용하려는 경우, Ultralytics로부터 공식 라이선스를 취득해야 합니다.
- 사용자 정의 가중치(weights)를 배포하려면 먼저`ultralytics`를 통해 호환 가능한 모델 아키텍처를 학습시켜야 합니다 (YOLOv8, YOLOv9 Det/Seg, YOLOv10 Det-only, YOLOv11, YOLOv12, YOLOv26 지원). 학습이 완료되면 내보낸`.pt`가중치 파일을`models/*`경로 내의 해당 하위 디렉터리에 넣으십시오.
- 소프트웨어에 자잘한 버그가 포함되어 있을 수 있습니다. 시간이 허락하는 대로 코드베이스를 지속적으로 최적화하고 흥미로운 기능을 추가할 예정입니다.
- 내보낸 추론 결과는`./run`경로 디렉터리 아래에 자동으로 저장됩니다.
- 핵심 UI 레이아웃 에셋 파일은`home.ui`입니다. Qt Designer를 사용하여 인터페이스를 수정하는 경우, 다음 명령을 사용하여 Python 소스 바인딩을 다시 생성하십시오:
```shell
pyside6-uic home.ui > ui/home.py
```
- 애플리케이션 그래픽 사전은`resources.qrc`입니다. 기본 앱 아이콘을 변경하는 경우, 다음 명령을 사용하여 에셋 파일을 다시 컴파일하십시오:
```shell
pyside6-rcc resources.qrc > ui/resources_rc.py
```
- 지향성 바운딩 박스(OBB) 기능은`Detect mode`에서 작동합니다. 사용자 정의`OBB`모델을 로드하려면 파일 이름에 하위 문자열`obb`가 명시적으로 포함되어야 합니다(예: yolov8n-obb.pt). 이름에`obb`가 포함되지 않은 파일은 표준 수평 탐지로 전환됩니다.

## 구현된 기능
### 1. 파이프라인 작업 선택
- 이미지 분류 (Classify)
- 객체 탐지 (Detect)
- 지향성 객체 탐지 (OBB)
- 포즈 추정 (Pose)
- 인스턴스 분할 (Segment)
- 객체 추적 (Track)
### 2. 스트림 및 컨텍스트 데이터 피드
- 단일 파일 미디어 평가 채널 (이미지 / 비디오)
- 디렉터리 일괄 파싱 및 평가 파이프라인
- 네이티브 UI 드래그 앤 드롭 파일 로드 처리기
- 통합 하드웨어 주변 장치 지원 (웹캠 / USB 카메라)
- `chose_rtsp`및`load_rtsp`함수 후크를 사용한 내장 라이브 스트림 바인딩

## 향후 로드맵
- [ ] 로컬 호스트 하드웨어 및 GPU 사용량 메트릭에 대한 실시간 추적을 구현합니다.
- [ ] 타임라인에 따른 동적 타겟 수를 표시하는 대화형 데이터 시각화 차트를 추가합니다.

## 참고 자료
- [PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [PySide6-YOLOv8](https://github.com/Jai-wei/YOLOv8-PySide6-GUI/tree/main)
- [YOLOSHOW](https://github.com/SwimmingLiu/YOLOSHOW/tree/31644373fca58aefcc9dba72a610c92031e5331b)
