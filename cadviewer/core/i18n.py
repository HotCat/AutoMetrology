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
    QTabWidget,
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
    "Light Source Control...": "光源控制...",
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
    "Run Dual-Light Measurement": "双光源测量",
    "Capture manual backlight/ring-light frames and evaluate using fixed-scale registration": "手动采集背光/环形光图像，并使用固定比例配准计算",
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
    "Force nearest line bias": "强制最近线偏置",
    "For stroke/window line pairs, use the stroke edge nearest the window edge": "对于印刷线/窗口线测量对，使用最靠近窗口边的印刷线边缘",
    "Line band:": "线条灰度带:",
    "+N band": "+N 灰度带",
    "-N band": "-N 灰度带",
    "Select which grayscale band to fit for printed lines. +N/-N use the CAD line normal from start to end; Auto preserves the existing CAD/pair-guided behavior.": "选择印刷线拟合的灰度带。+N/-N 使用 CAD 线从起点到终点方向的法线；自动模式保留现有的 CAD/测量对引导行为。",
    "Line ID": "线 ID",
    "Band": "灰度带",
    "Optional per-line band overrides. Line ID may be a full ID, DXF handle, or unique prefix.": "可选的单线灰度带覆盖。线 ID 可以是完整 ID、DXF 句柄或唯一前缀。",
    "Use Selected Line": "使用所选线",
    "Add or update the currently selected CAD line using the selected row band.": "用当前行的灰度带添加或更新所选 CAD 线。",
    "Add Row": "添加行",
    "Remove": "删除",
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
    "Apply correction map": "应用校正图",
    "Apply saved residual/coordinate correction in measurement. Turn off to compare affine-only behavior.": "在测量中应用已保存的残差/坐标校正。关闭后可对比仅使用仿射的效果。",
    "Correction map enabled; rerun registration/evaluate to compare.": "校正图已启用；请重新配准/计算以对比。",
    "Correction map disabled; rerun registration/evaluate to compare.": "校正图已禁用；请重新配准/计算以对比。",
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
    "Window Register": "窗口配准",
    "Window CAD Edges": "窗口 CAD 边",
    "Detect:": "检测:",
    "Fraction:": "比例:",
    "Dark window": "暗窗口",
    "Bright backlight": "明亮背光",
    "Printed grid": "印刷网格",
    "Select CAD edge, click Add; need 4": "选择 CAD 边后点击添加；需要 4 条",
    "Add": "添加",
    "Camera Capture": "相机采集",
    "Refresh": "刷新",
    "Open": "打开",
    "Close": "关闭",
    "Capture Frame": "采集图像",
    "Focus Preview": "对焦预览",
    "Save Live": "保存预览",
    "Save Backlight": "保存背光",
    "Save Ring": "保存环形光",
    "Test Backlight Capture": "测试背光采集",
    "Test Ring-Light Capture": "测试环形光采集",
    "Settle ms:": "稳定 ms:",
    "Debug Window Registration": "调试窗口配准",
    "Backlight Ready": "背光已准备",
    "Ring-Light Ready": "环形光已准备",
    "Please turn ON backlight, turn OFF ring light, then click Confirm.": "请打开背光、关闭环形光，然后点击确认。",
    "Please turn OFF backlight and turn ON ring light, then click Confirm.": "请关闭背光、打开环形光，然后点击确认。",
    "Confirm": "确认",
    "Cancel": "取消",
    "Dual-light capture cancelled": "双光源采集已取消",
    "Dual-light measurement failed: pixel size calibration missing": "双光源测量失败：缺少像素尺寸标定",
    "Dual-light measurement failed: select 4 window CAD edges": "双光源测量失败：请选择 4 条窗口 CAD 边",
    "Dual-light measurement: capture backlight and ring-light frames...": "双光源测量：正在采集背光和环形光图像...",
    "Dual-light capture failed:": "双光源采集失败:",
    "Dual-light measurement cancelled": "双光源测量已取消",
    "Dual-Light Measurement": "双光源测量",
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
    "Light Source Control": "光源控制",
    "Controller Connection": "控制器连接",
    "Device:": "设备:",
    "Baud:": "波特率:",
    "Timeout:": "超时:",
    "Backlight settle delay:": "背光稳定延时:",
    "Ring-light settle delay:": "环形光稳定延时:",
    "Ring Light CH1": "环形光 CH1",
    "Ring Light CH2": "环形光 CH2",
    "Backlight CH4": "背光 CH4",
    "Output:": "输出:",
    "Brightness:": "亮度:",
    "On": "开",
    "Off": "关",
    "Apply All": "全部应用",
    "All Off": "全部关闭",
    "Read Brightness": "读取亮度",
    "Switch this controller channel output on or off": "打开或关闭该控制器通道输出",
    "Controller OK.": "控制器正常。",
    "Light settings applied": "光源设置已应用",
    "All configured light channels are off": "所有已配置光源通道已关闭",
    "Light controller connection failed:": "光源控制器连接失败:",
    "Read brightness failed:": "读取亮度失败:",
    "Apply light settings failed:": "应用光源设置失败:",
    "Turning lights off failed:": "关闭光源失败:",
    "Dual-light capture: backlight on, ring light off": "双光源采集：背光开启，环形光关闭",
    "Dual-light capture: ring light on, backlight off": "双光源采集：环形光开启，背光关闭",
    "Checking converters...": "正在检测转换器...",
    "Browse...": "浏览...",
    "Test Connection": "测试连接",
    "ODA detected": "已检测到 ODA",
    "dwg2dxf detected": "已检测到 dwg2dxf",
    "No converter found": "未找到转换器",
    "No converter available": "无可用转换器",
    "Converter OK": "转换器正常",
    "detected": "已检测到",
    "OK — executable works": "正常 - 可执行文件可用",
    "Failed — not a valid executable": "失败 - 不是有效的可执行文件",
    "DWG converter not found — configure in Settings": "未找到 DWG 转换器 - 请在设置中配置",
    "ODA File Converter not found — install and configure via Settings menu": "未找到 ODA File Converter - 请安装并在设置菜单中配置",
    "Load Telecentric Image": "加载远心图像",
    "Image Source": "图像来源",
    "Select PNG, BMP, or TIF file...": "选择 PNG、BMP 或 TIF 文件...",
    "Camera preview": "相机预览",
    "Capture": "采集",
    "No image selected": "未选择图像",
    "Ignore saved lens calibration for this image": "此图像忽略已保存的镜头标定",
    "Load or capture this image without applying the saved camera/lens calibration": "加载或采集此图像时不应用已保存的相机/镜头标定",
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
    "Photo of printed chessboard pattern...": "印刷棋盘格图像...",
    "Calibrate Pixel Size": "标定像素尺寸",
    "Compute Mount Angles": "计算安装角度",
    "Select Chessboard Image": "选择棋盘格图像",
    "Select a chessboard image first.": "请先选择棋盘格图像。",
    "Cannot read image file.": "无法读取图像文件。",
    "No frame available.": "没有可用图像。",
    "Error: OpenCV not available": "错误：OpenCV 不可用",
    "Calibrate pixel size first so chessboard corners are available.": "请先标定像素尺寸，以获得棋盘格角点。",
    "Run and save Lens Calibration first. Mount angles require camera intrinsics.": "请先运行并保存镜头标定。安装角度计算需要相机内参。",
    "Camera pose estimation failed.": "相机位姿估计失败。",
    "From Camera": "来自相机",
    "From Files": "来自文件",
    "Camera not streaming": "相机未推流",
    "Waiting for camera...": "等待相机...",
    "Load images from files": "从文件加载图像",
    "Add Files...": "添加文件...",
    "Clear All": "全部清除",
    "Reload Saved Set": "重新加载已保存图像集",
    "Open Folder": "打开文件夹",
    "Collected Images": "已采集图像",
    "Images: 0 | Corners detected: 0": "图像: 0 | 检测到角点: 0",
    "Remove Selected": "删除所选",
    "Model:": "模型:",
    "Standard": "标准",
    "Rational": "有理模型",
    "Rational + Thin Prism": "有理模型 + 薄棱镜",
    "Rational + Thin Prism + Tilted": "有理模型 + 薄棱镜 + 倾斜",
    "Run Calibration": "运行标定",
    "Compare Models": "比较模型",
    "Results": "结果",
    "No calibration results yet.": "暂无标定结果。",
    "Save to Config": "保存到配置",
    "Pixel Size Calibration": "像素尺寸标定",
    "Lens Calibration": "镜头标定",
    "No frame available to capture.": "没有可采集的图像。",
    "All images cleared.": "已清除所有图像。",
    "Another calibration task is already running.": "另一个标定任务正在运行。",
    "No saved calibration images found.": "未找到已保存的标定图像。",
    "Error: OpenCV not available.": "错误：OpenCV 不可用。",
    "Calibration failed.": "标定失败。",
    "Model comparison failed.": "模型比较失败。",
    "Model comparison complete.": "模型比较完成。",
    "No detected calibration images to save.": "没有可保存的已检测标定图像。",
    "Calibration saved to configuration.": "标定已保存到配置。",
    "Select a CAD line first": "请先选择 CAD 线",
    "Selected feature is not a line": "所选特征不是线",
    "Select a CAD line edge first.": "请先选择 CAD 线边。",
    "Window edge already added.": "窗口边已添加。",
    "Window edge list already has 4 lines; clear it first.": "窗口边列表已有 4 条线；请先清除。",
    "Window CAD edges cleared.": "窗口 CAD 边已清除。",
    "Teach complete. Click 'Save Pose Template' to store.": "示教完成。点击“保存位姿模板”进行保存。",
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


def _translate_tabs(tabs: QTabWidget) -> None:
    keys = tabs.property("i18n_tab_texts")
    if keys is None:
        keys = [tabs.tabText(i) for i in range(tabs.count())]
        tabs.setProperty("i18n_tab_texts", keys)
    for i, key in enumerate(list(keys)[: tabs.count()]):
        tabs.setTabText(i, tr(str(key)))


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
        if isinstance(obj, QTabWidget):
            _translate_tabs(obj)
