"""Runtime UI translation helpers.

The application uses English source strings as stable translation keys.  Widgets
store their original English text in Qt dynamic properties the first time they
are translated, so switching language at runtime is reversible without
rebuilding the UI.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDialog,
    QDockWidget,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QTableWidget,
    QTextEdit,
    QTreeWidget,
    QWidget,
)


LANG_EN = "en"
LANG_ZH_CN = "zh_CN"
SUPPORTED_LANGUAGES = (LANG_EN, LANG_ZH_CN)


ZH_CN: dict[str, str] = {
    "English": "English",
    "CAD Inspection Tool — Metrology DXF Viewer": "CAD 检测工具 - 计量 DXF 查看器",
    "Registration": "配准",
    "Features": "特征",
    "Evaluated": "已计算",
    "Errors": "错误",
    "No Measurement": "无测量",
    "No measurement": "无测量",
    "records": "记录",
    "Simplified Chinese": "简体中文",
    "Language": "语言",
    "CAD Inspection Tool - Metrology DXF Viewer": "CAD 检测工具 - 计量 DXF 查看器",
    "Open DXF": "打开 DXF",
    "Open DXF...": "打开 DXF...",
    "Import DWG": "导入 DWG",
    "Import DWG File": "导入 DWG 文件",
    "DWG Converter Not Found": "未找到 DWG 转换器",
    "No DWG converter is installed.": "未安装 DWG 转换器。",
    "Image loaded. Ready for registration.": "图像已加载，可以配准。",
    "Fiducial ROIs updated from image picker": "已从图像选择器更新基准点 ROI",
    "Loading": "正在加载",
    "Loaded": "已加载",
    "features from": "个特征，来自",
    "Production cycle: capturing camera frame...": "生产流程：正在采集相机图像...",
    "Production cycle failed during camera capture": "生产流程在相机采集阶段失败",
    "Production cycle: applying auto registration...": "生产流程：正在执行自动配准...",
    "Production cycle failed during auto registration": "生产流程在自动配准阶段失败",
    "Production cycle: evaluating measurement queries...": "生产流程：正在计算测量查询...",
    "Measurement query pair selection cancelled": "测量查询配对选择已取消",
    "Production profile name is empty": "生产参数组名称为空",
    "Camera support is not available": "相机支持不可用",
    "Camera is not open": "相机未打开",
    "Canvas is not available": "画布不可用",
    "No frame to capture": "没有可采集的图像",
    "Select a CAD circle first": "请先选择 CAD 圆",
    "Selected CAD feature is not a circle": "所选 CAD 特征不是圆",
    "Lens calibration applied to registration image": "已对配准图像应用镜头标定",
    "Error: pipeline not initialized": "错误：流程未初始化",
    "No anchor candidates found": "未找到候选锚点",
    "Teach points cleared": "示教点已清除",
    "Error: need 2 CAD + 2 image points": "错误：需要 2 个 CAD 点和 2 个图像点",
    "Import DWG...": "导入 DWG...",
    "Fit All": "适配全部",
    "Pan": "平移",
    "Select": "选择",
    "File": "文件",
    "Exit": "退出",
    "View": "视图",
    "Registration Panel": "配准面板",
    "Measurement Window": "测量窗口",
    "Settings": "设置",
    "Configure DWG Converter...": "配置 DWG 转换器...",
    "Camera Calibration...": "相机标定...",
    "Help": "帮助",
    "About": "关于",
    "Ready — Open a DXF file to begin inspection": "就绪 - 打开 DXF 文件开始检测",
    "Ready - Open a DXF file to begin inspection": "就绪 - 打开 DXF 文件开始检测",
    "Ready": "就绪",
    "Features: 0": "特征: 0",
    "Measurement Queries": "测量查询",
    "Load": "加载",
    "Save": "保存",
    "Run Production": "生产运行",
    "Evaluate": "计算",
    "Export Results": "导出结果",
    "View Logs": "查看日志",
    "Show production measurement logs": "显示生产测量日志",
    "Return to live measurement queries": "返回实时测量查询",
    "Capture camera frame, register, and evaluate queries (F5)": "采集相机图像、配准并计算查询 (F5)",
    "Pick Lines Pair": "选择直线对",
    "Pick Circles Pair": "选择圆对",
    "Pick Circle": "选择圆",
    "Pick Arc": "选择圆弧",
    "Cancel Pick": "取消选择",
    "Pair picker idle": "选择器空闲",
    "Picking lines": "正在选择直线",
    "Picking circles": "正在选择圆",
    "Picking circle": "正在选择圆",
    "Picking arc": "正在选择圆弧",
    "Tol %:": "公差 %:",
    "Tolerance percent used when generated queries are added": "生成查询时使用的百分比公差",
    "No queries evaluated": "尚未计算查询",
    "Query": "查询",
    "Value": "测量值",
    "Nominal": "名义值",
    "Deviation": "偏差",
    "Threshold": "阈值",
    "Status": "状态",
    "Production Log Viewer": "生产日志查看器",
    "No production records": "无生产记录",
    "Daily Records": "每日记录",
    "Status / Time": "状态 / 时间",
    "CAD": "CAD",
    "Image": "图像",
    "Rows": "行数",
    "Select a production record": "选择生产记录",
    "No production records for selected day": "所选日期无生产记录",
    "OK": "合格",
    "NG": "不合格",
    "Auto Registration": "自动配准",
    "Production Parameters": "生产参数",
    "Profile:": "参数组:",
    "Save As...": "另存为...",
    "Delete": "删除",
    "Saves camera settings, fiducials, and ROIs.": "保存相机设置、基准点和 ROI。",
    "Image Registration": "图像配准",
    "Load Image...": "加载图像...",
    "No image loaded": "未加载图像",
    "Method:": "方法:",
    "Full Silhouette": "完整轮廓",
    "Convex Hull (partial FOV)": "凸包 (局部视野)",
    "Fiducial-Based": "基准点",
    "Teach + ICP": "示教 + ICP",
    "Anchors:": "锚点:",
    "DXF handles, e.g. 120C3,12121": "DXF 句柄，例如 120C3,12121",
    "Auto": "自动",
    "Coarse Registration": "粗配准",
    "Refine (Contour ICP)": "精配准 (轮廓 ICP)",
    "Full Registration": "完整配准",
    "Teach Initial Pose": "示教初始位姿",
    "Save Pose Template": "保存位姿模板",
    "Clear": "清除",
    "Auto 2-Point Correspondence": "自动两点对应",
    "CAD P1:": "CAD P1:",
    "CAD P2:": "CAD P2:",
    "Select CAD circle, click Use": "选择 CAD 圆后点击使用",
    "Use": "使用",
    "ROI P1:": "ROI P1:",
    "ROI P2:": "ROI P2:",
    "x,y,w,h": "x,y,w,h",
    "Pick ROIs...": "选择 ROI...",
    "Auto Register": "自动配准",
    "Camera Capture": "相机采集",
    "Refresh": "刷新",
    "Open": "打开",
    "Close": "关闭",
    "Capture Frame": "采集图像",
    "Focus Preview": "对焦预览",
    "No camera connected": "未连接相机",
    "No camera detected": "未检测到相机",
    "Properties": "属性",
    "Select a feature to view properties": "选择特征以查看属性",
    "General": "常规",
    "Geometry": "几何",
    "Measurement": "测量",
    "Feature Browser": "特征浏览器",
    "Filter features...": "过滤特征...",
    "Configure DWG Converter": "配置 DWG 转换器",
    "Checking converters...": "正在检测转换器...",
    "Browse...": "浏览...",
    "Test Connection": "测试连接",
    "Load Telecentric Image": "加载远心图像",
    "Image Source": "图像来源",
    "Select PNG, BMP, or TIF file...": "选择 PNG、BMP 或 TIF 文件...",
    "Camera preview": "相机预览",
    "Capture": "采集",
    "No image selected": "未选择图像",
    "Pixel Size": "像素尺寸",
    "Select Fiducial Search ROIs": "选择基准点搜索 ROI",
    "Draw ROI P1": "绘制 ROI P1",
    "Draw ROI P2": "绘制 ROI P2",
    "Clear Active": "清除当前",
    "Drag on the image to draw the active ROI. Existing boxes show saved search areas.": "在图像上拖拽以绘制当前 ROI。已有矩形表示保存的搜索区域。",
    "Import DWG": "导入 DWG",
    "Preparing conversion...": "正在准备转换...",
    "Conversion failed": "转换失败",
    "Conversion complete": "转换完成",
    "Camera Live Preview": "相机实时预览",
    "Fit to Window": "适配窗口",
    "Waiting for frames...": "等待图像...",
    "Live": "实时",
    "Camera Settings": "相机设置",
    "Camera closed": "相机已关闭",
    "Camera Calibration": "相机标定",
    "Chessboard Pattern": "棋盘格",
    "Cols:": "列:",
    "Rows:": "行:",
    "Cell:": "单元:",
    "Calibrate Pixel Size": "标定像素尺寸",
    "Collected Images": "已采集图像",
    "Remove Selected": "删除所选",
    "Run Calibration": "运行标定",
    "Results": "结果",
    "No calibration results yet.": "暂无标定结果。",
    "Save to Config": "保存到配置",
}


class I18nManager(QObject):
    """Holds current UI language and translates English source strings."""

    language_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._language = LANG_EN

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            language = LANG_EN
        if language == self._language:
            return
        self._language = language
        self.language_changed.emit(language)

    def tr(self, text: str) -> str:
        if self._language == LANG_ZH_CN:
            return ZH_CN.get(text, text)
        return text


i18n = I18nManager()


def tr(text: str) -> str:
    return i18n.tr(text)


def set_language(language: str) -> None:
    i18n.set_language(language)


def _remember(obj: QObject, prop: str, value):
    existing = obj.property(prop)
    if existing is None:
        obj.setProperty(prop, value)
        return value
    return existing


def _translate_text_prop(obj: QObject, getter: str, setter: str, prop: str) -> None:
    value = getattr(obj, getter)()
    if not isinstance(value, str) or value == "":
        return
    key = _remember(obj, prop, value)
    getattr(obj, setter)(tr(str(key)))


def _translate_headers(widget: QTableWidget | QTreeWidget) -> None:
    count = widget.columnCount()
    if count <= 0:
        return
    keys = widget.property("i18n_horizontal_headers")
    if keys is None:
        keys = []
        if isinstance(widget, QTreeWidget):
            header = widget.headerItem()
            for col in range(count):
                keys.append(header.text(col) if header is not None else "")
        else:
            for col in range(count):
                item = widget.horizontalHeaderItem(col)
                keys.append(item.text() if item is not None else "")
        widget.setProperty("i18n_horizontal_headers", keys)
    if isinstance(widget, QTreeWidget):
        header = widget.headerItem()
        if header is not None:
            for col, key in enumerate(list(keys)[:count]):
                header.setText(col, tr(str(key)))
        return
    for col, key in enumerate(list(keys)[:count]):
        item = widget.horizontalHeaderItem(col)
        if item is not None:
            item.setText(tr(str(key)))


def _translate_combo(combo: QComboBox) -> None:
    keys = combo.property("i18n_items")
    if keys is None:
        keys = [combo.itemText(i) for i in range(combo.count())]
        combo.setProperty("i18n_items", keys)
    for i, key in enumerate(list(keys)[: combo.count()]):
        combo.setItemText(i, tr(str(key)))


def _objects(root: QObject) -> Iterable[QObject]:
    yield root
    yield from root.findChildren(QObject)


def retranslate_widget_tree(root: QObject) -> None:
    """Retranslate common Qt widgets/actions under ``root`` in place."""
    for obj in _objects(root):
        if isinstance(obj, QAction):
            _translate_text_prop(obj, "text", "setText", "i18n_text")
            _translate_text_prop(obj, "toolTip", "setToolTip", "i18n_tooltip")
            continue
        if isinstance(obj, QMenu):
            _translate_text_prop(obj, "title", "setTitle", "i18n_title")
        if isinstance(obj, QMainWindow | QDialog | QDockWidget | QWidget):
            _translate_text_prop(obj, "windowTitle", "setWindowTitle", "i18n_window_title")
        if isinstance(obj, QLabel | QAbstractButton):
            _translate_text_prop(obj, "text", "setText", "i18n_text")
        if isinstance(obj, QGroupBox):
            _translate_text_prop(obj, "title", "setTitle", "i18n_title")
        if isinstance(obj, QLineEdit | QTextEdit):
            _translate_text_prop(
                obj, "placeholderText", "setPlaceholderText", "i18n_placeholder"
            )
        if isinstance(obj, QTableWidget | QTreeWidget):
            _translate_headers(obj)
        if isinstance(obj, QComboBox):
            _translate_combo(obj)
