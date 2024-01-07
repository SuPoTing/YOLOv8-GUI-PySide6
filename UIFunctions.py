from main import *
from custom_grips import CustomGrip
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QEvent, QTimer
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import time

GLOBAL_STATE = False    # max min flag
GLOBAL_TITLE_BAR = True


class UIFuncitons(MainWindow):
    # 展開左側菜單
    def toggleMenu(self, enable):
        if enable:
            standard = 68  # 左側菜單的標準寬度
            maxExtend = 180  # 左側菜單展開時的最大寬度
            width = self.LeftMenuBg.width()  # 現在的菜單寬度

            if width == 68:  # 如果菜單目前是縮起來的
                widthExtended = maxExtend  # 展開後的寬度
            else:
                widthExtended = standard  # 收縮後的寬度

            # 動畫效果
            self.animation = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.animation.setDuration(500)  # 動畫時間（毫秒）
            self.animation.setStartValue(width)  # 動畫的起始寬度
            self.animation.setEndValue(widthExtended)  # 動畫的結束寬度
            self.animation.setEasingCurve(QEasingCurve.InOutQuint)  # 動畫的緩動曲線
            self.animation.start()  # 開始執行動畫

    # 展開右側的設定選單
    def settingBox(self, enable):
        if enable:
            # 獲取寬度
            widthRightBox = self.prm_page.width()  # 右側設定選單的寬度
            widthLeftBox = self.LeftMenuBg.width()  # 左側菜單的寬度
            maxExtend = 220  # 設定選單展開時的最大寬度
            standard = 0

            # 設定最大寬度
            if widthRightBox == 0:  # 如果右側設定選單目前是收縮的
                widthExtended = maxExtend  # 展開後的寬度
            else:
                widthExtended = standard  # 收縮後的寬度

            # 設定左側菜單的動畫
            self.left_box = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.left_box.setDuration(500)  # 動畫時間（毫秒）
            self.left_box.setStartValue(widthLeftBox)  # 動畫的起始寬度
            self.left_box.setEndValue(68)  # 動畫的結束寬度（收縮的寬度）
            self.left_box.setEasingCurve(QEasingCurve.InOutQuart)  # 動畫的緩動曲線

            # 設定右側設定選單的動畫
            self.right_box = QPropertyAnimation(self.prm_page, b"minimumWidth")
            self.right_box.setDuration(500)  # 動畫時間（毫秒）
            self.right_box.setStartValue(widthRightBox)  # 動畫的起始寬度
            self.right_box.setEndValue(widthExtended)  # 動畫的結束寬度
            self.right_box.setEasingCurve(QEasingCurve.InOutQuart)  # 動畫的緩動曲線

            # 創建一個平行動畫組
            self.group = QParallelAnimationGroup()
            self.group.addAnimation(self.left_box)
            self.group.addAnimation(self.right_box)
            self.group.start()  # 開始執行動畫

    # 展開右側的設定選單
    def cam_settingBox(self, enable):
        if enable:
            # 獲取寬度
            widthRightBox = self.prm_page_cam.width()  # 右側設定選單的寬度
            widthLeftBox = self.LeftMenuBg.width()  # 左側菜單的寬度
            maxExtend = 220  # 設定選單展開時的最大寬度
            standard = 0

            # 設定最大寬度
            if widthRightBox == 0:  # 如果右側設定選單目前是收縮的
                widthExtended = maxExtend  # 展開後的寬度
            else:
                widthExtended = standard  # 收縮後的寬度

            # 設定左側菜單的動畫
            self.left_box = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.left_box.setDuration(500)  # 動畫時間（毫秒）
            self.left_box.setStartValue(widthLeftBox)  # 動畫的起始寬度
            self.left_box.setEndValue(68)  # 動畫的結束寬度（收縮的寬度）
            self.left_box.setEasingCurve(QEasingCurve.InOutQuart)  # 動畫的緩動曲線

            # 設定右側設定選單的動畫
            self.right_box = QPropertyAnimation(self.prm_page_cam, b"minimumWidth")
            self.right_box.setDuration(500)  # 動畫時間（毫秒）
            self.right_box.setStartValue(widthRightBox)  # 動畫的起始寬度
            self.right_box.setEndValue(widthExtended)  # 動畫的結束寬度
            self.right_box.setEasingCurve(QEasingCurve.InOutQuart)  # 動畫的緩動曲線

            # 創建一個平行動畫組
            self.group = QParallelAnimationGroup()
            self.group.addAnimation(self.left_box)
            self.group.addAnimation(self.right_box)
            self.group.start()  # 開始執行動畫

    # 最大化/還原視窗
    def maximize_restore(self):
        global GLOBAL_STATE  # 使用全局變數
        status = GLOBAL_STATE  # 取得全局變數的值
        if status == False:  # 如果視窗不是最大化狀態
            GLOBAL_STATE = True  # 設置全局變數為 True（最大化狀態）
            self.showMaximized()  # 最大化視窗
            self.max_sf.setToolTip("Restore")  # 更改最大化按鈕的提示文本
            self.frame_size_grip.hide()  # 隱藏視窗大小調整按鈕
            self.left_grip.hide()  # 隱藏四邊調整的按鈕
            self.right_grip.hide()
            self.top_grip.hide()
            self.bottom_grip.hide()
        else:
            GLOBAL_STATE = False  # 設置全局變數為 False（非最大化狀態）
            self.showNormal()  # 還原視窗（最小化）
            self.resize(self.width() + 1, self.height() + 1)  # 修復最小化後的視窗大小
            self.max_sf.setToolTip("Maximize")  # 更改最大化按鈕的提示文本
            self.frame_size_grip.show()  # 顯示視窗大小調整按鈕
            self.left_grip.show()  # 顯示四邊調整的按鈕
            self.right_grip.show() # 顯示四邊調整的按鈕
            self.top_grip.show() # 顯示四邊調整的按鈕
            self.bottom_grip.show() # 顯示四邊調整的按鈕
    
    # 視窗控制的定義
    def uiDefinitions(self):
        # 雙擊標題欄最大化/還原
        def dobleClickMaximizeRestore(event):
            if event.type() == QEvent.MouseButtonDblClick:
                QTimer.singleShot(250, lambda: UIFuncitons.maximize_restore(self))
        self.top.mouseDoubleClickEvent = dobleClickMaximizeRestore
        
        # 移動視窗 / 最大化 / 還原
        def moveWindow(event):
            if GLOBAL_STATE:  # 如果視窗已最大化，則切換到還原狀態
                UIFuncitons.maximize_restore(self)
            if event.buttons() == Qt.LeftButton:  # 移動視窗
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
        self.top.mouseMoveEvent = moveWindow
        
        # 自定義拉伸按鈕
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)

        # 最小化視窗
        self.min_sf.clicked.connect(lambda: self.showMinimized())
        # 最大化/還原視窗
        self.max_sf.clicked.connect(lambda: UIFuncitons.maximize_restore(self))
        # 關閉應用程式
        self.close_button.clicked.connect(self.close)

    # 控制視窗四邊的拉伸
    def resize_grips(self):
        # 設置左側拉伸按鈕的位置和大小
        self.left_grip.setGeometry(0, 10, 10, self.height())
        # 設置右側拉伸按鈕的位置和大小
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        # 設置上側拉伸按鈕的位置和大小
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        # 設置下側拉伸按鈕的位置和大小
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # 顯示模組以添加陰影效果
    def shadow_style(self, widget, Color):
        shadow = QGraphicsDropShadowEffect(self)  # 創建陰影效果對象
        shadow.setOffset(8, 8)  # 設定陰影的偏移量
        shadow.setBlurRadius(38)  # 設定陰影的模糊半徑
        shadow.setColor(Color)  # 設定陰影的顏色
        widget.setGraphicsEffect(shadow)  # 將陰影效果應用到指定的小部件