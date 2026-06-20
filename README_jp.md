# Ultralytics-Pyside6-GUI `V2.5` a GUI for Ultralytics 8.4.70
---
  <p align="center"> 
  <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README.md">English</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-tw.md">繁體中文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_zh-cn.md">简体中文</a> &nbsp; | &nbsp; 日文</a> &nbsp; | &nbsp; <a href="https://github.com/SuPoTing/Ultralytics-GUI-PySide6/blob/v2.5/README_kr.md">한국어</a>
 </p>

![](./img/preview.png)

## 環境構築
### 1. 仮想環境の構築
`create_env.bat`をクリックしてPython 3.10の仮想環境を構築し、環境をアクティベートします。

### 2. アプリケーションの実行
`main.bat`をクリックしてアプリケーションを起動します。

## PyInstaller によるパッケージング (Auto-Py-To-Exe 経由)
### 1. 仮想環境の構築
`create_env.bat`をクリックしてPython 3.10の仮想環境を生成し、次に`activate.bat`をクリックしてアクティベートします。

### 2. auto-py-to-exe GUI 画面の起動
```shell
auto-py-to-exe
```

### 3. スクリプトパスと追加ファイルの追加
Script Location:
```shell
(your YOLOv8-GUI-PySide6-main PATH)\main.py
```

Additional Files / Folders:
「Add Folder」をクリックし、以下を選択します：
```shell
(your YOLOv8-GUI-PySide6-main PATH)\venv\Lib\site-packages\ultralytics
```

「Convert .py to .exe」をクリックします。

### 4. 必須ディレクトリのコピー
`config`、`img`、`models`、`ui`、`utils`フォルダを以下にコピーします：
```shell
(your YOLOv8-GUI-PySide6-main PATH)\output\main
```

### 5. main.exeの実行
出力ディレクトリ内の`main.exe`を実行して、コンパイルされたアプリケーションを起動します。

## Nuitka によるパッケージング
### 1. 仮想環境の構築
`create_env-nuitka.bat`をクリックしてPython 3.10の仮想環境を生成し、次に`activate.bat`をクリックしてアクティベートします。

### 2. Nuitkaコンパイルコマンドの実行
```shell
nuitka --standalone --msvc=latest --lto=yes --enable-plugin=pyside6 --module-parameter=torch-disable-jit=no --include-package=ultralytics main.py
```
### 3. アセットフォルダのコピー
.\venv\Lib\site-packages\ultralytics フォルダを、新しく生成された`.dist`ディレクトリ内にコピーします。

### 4. main.exe の実行
`.dist`ディレクトリに移動し、`main.exe`を実行してアプリケーションを起動します。

## 重要な注意事項
- `ultralytics`は`AGPL-3.0`ライセンスの下で提供されています。商用目的で使用する場合は、Ultralytics から正式なライセンスを取得する必要があります。
- 独自のカスタム重みを配備するには、まず`ultralytics`を介して互換性のあるモデルアーキテクチャをトレーニングする必要があります（YOLOv8、YOLOv9 Det/Seg、YOLOv10 Det-only、YOLOv11、YOLOv12、YOLOv26 に対応）。トレーニング完了後、エクスポートした`.pt`重みファイルを`models/*`パス内の対応するサブディレクトリに配置してください。
- ソフトウェアには軽微なバグが含まれている可能性があります。時間の経過とともに、コードベースの最適化や興味深い機能の追加を継続して行っていきます。
- エクスポートされた推論結果は、`./run`パスディレクトリの下に自動的に保存されます。
- コア UI レイアウトのアセットファイルは`home.ui`です。Qt Designer を使用してインターフェースを変更した場合は、次のコマンドでPythonソースバインディングを再生成してください：
```shell
pyside6-uic home.ui > ui/home.py
```
- アプリケーションのグラフィック辞書は`resources.qrc`です。デフォルトのアプリアイコンを変更した場合は、次のコマンドを使用してアセットファイルを再コンパイルしてください：
```shell
pyside6-rcc resources.qrc > ui/resources_rc.py
```
- 指向性バウンディングボックス（OBB）機能は`Detect mode`の下で動作します。カスタム`OBB`モデルをロードするには、ファイル名に明示的に文字列 obb が含まれている必要があります（例：yolov8n-obb.pt）。名前に`obb`が含まれていないファイルは、標準の水平検出にフォールバックされます。

## 実装済み機能
### 1. パイプラインタスクの選択
- 画像分類 (Classify)
- 物体検出 (Detect)
- 指向性物体検出 (OBB)
- 姿勢推計 (Pose)
- インスタンスセグメンテーション (Segment)
- 物体追跡 (Track)
### 2. ストリーム＆コンテキストデータフィード
- 単一メディアファイル評価チャネル（画像 / 動画）
- ディレクトリ一括解析および評価パイプライン
- ネイティブ UI ドラッグ＆ドロップ取り込みファイルハンドラー
- 統合されたハードウェア周辺機器のサポート（Webカメラ / USBカメラ）
- `chose_rtsp`および`load_rtsp`関数フックを使用した内蔵ライブストリームバインディング

## 今後のロードマップ
- [ ] ローカルホストのハードウェアおよび GPU 使用率メトリクスのライブトラッキングを実装する。
- [ ] タイムライン上の動的なターゲットカウントを表示するインタラクティブなデータ可視化チャートを追加する。

## 参考文献
- PyQt5-YOLOv5
- ultralytics
- PySide6-YOLOv8
- YOLOSHOW
