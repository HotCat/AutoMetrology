# Windows 10 部署指南

以下步骤用于在全新的 Windows 10 64 位工作站上从 GitHub 克隆 AutoMetrology，并完成运行、相机 SDK 配置和 PyInstaller 打包。

本文档对应仓库：`https://github.com/HotCat/AutoMetrology.git`

---

## 1. 准备 Windows 10 工作站

建议环境：

- Windows 10 64 位
- 至少 8 GB 内存
- 管理员权限（安装相机 SDK、VC++ 运行时和 ODA 转换器时需要）
- 可访问 GitHub 和 Python/PyPI 镜像
- MindVision 工业相机（如果需要生产采图）

---

## 2. 安装 Git

1. 下载 Git for Windows：
   - https://git-scm.com/download/win
2. 默认选项安装即可。
3. 打开 **PowerShell** 或 **Anaconda Prompt**，验证：

```bat
git --version
```

---

## 3. 安装 Miniconda

1. 下载 Miniconda Windows x64：
   - 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
   - 官方：https://docs.conda.io/en/latest/miniconda.html
2. 运行安装程序。
3. 建议勾选：
   - `Add Miniconda to my PATH environment variable`
4. 安装完成后打开 **Anaconda Prompt**，验证：

```bat
conda --version
python --version
```

---

## 4. 克隆代码

选择一个工作目录，例如 `C:\Projects`：

```bat
cd /d C:\Projects
git clone https://github.com/HotCat/AutoMetrology.git
cd AutoMetrology
```

如果网络慢，可以配置 Git 代理或使用公司内部镜像。

---

## 5. 创建 Python 环境

推荐 Python 3.11：

```bat
conda create -n autometrology python=3.11 -y
conda activate autometrology
python --version
```

以后每次运行或打包前都要先执行：

```bat
conda activate autometrology
```

---

## 6. 安装项目依赖

仓库包含 Windows 专用依赖文件：

```bat
pip install -r requirements-windows.txt
```

如果下载慢，使用清华 PyPI 镜像：

```bat
pip install -r requirements-windows.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证关键依赖：

```bat
python -c "import PySide6, cv2, ezdxf, scipy; print('basic deps ok')"
python -c "import diplib; print('diplib ok')"
```

> `diplib` 用于更稳定的基准圆/圆弧检测。如果 `diplib` 安装失败，软件仍可启动，但部分自动检测会退回 OpenCV 路径，精度和稳定性可能下降。生产部署建议安装成功后再打包。

---

## 7. 安装 MindVision 相机 SDK

如果只做 DXF/图片离线测试，可跳过本节。生产相机采图必须安装 SDK。

1. 从 MindVision 官网下载 Windows SDK：
   - http://www.mindvision.com.cn/
   - 通常路径：技术支持 → SDK 下载
2. 安装 `MVCAMSDK.exe`。
3. 默认安装目录通常为：

```bat
C:\Program Files\MindVision\MVCAMSDK
```

4. 检查 64 位 Runtime DLL：

```bat
dir "C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64\MVCAMSDK_X64.dll"
```

5. 将 Runtime 路径加入系统 `PATH`：

```text
C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64
```

操作方法：

1. 右键“此电脑” → 属性。
2. 高级系统设置 → 环境变量。
3. 在“系统变量”中编辑 `Path`。
4. 新增上面的 Runtime 路径。
5. 保存后重新打开 Anaconda Prompt。

验证 SDK 可被 Python 加载：

```bat
python -c "from cadviewer.camera import HAS_CAMERA; print(HAS_CAMERA)"
```

输出 `True` 表示 SDK wrapper 已正常加载。没有连接相机时，程序仍可能显示未检测到设备，这是正常状态。

---

## 8. （可选）安装 ODA File Converter

软件可以导入 DXF。若需要导入 DWG，需额外安装 ODA File Converter。

1. 下载 ODA File Converter：
   - https://www.opendesign.com/guestfiles/oda_file_converter
2. 默认安装路径通常为：

```bat
C:\Program Files\ODA\ODAFileConverter
```

3. 软件会自动搜索常见安装路径。也可以在软件菜单中配置转换器路径。

> PyInstaller 包不会内置 ODA File Converter。目标机器需要单独安装。

---

## 9. 安装 VC++ 运行时

如果运行或打包后启动时报 `vcruntime140.dll`、`msvcp140.dll`、OpenCV DLL 加载失败等错误，安装 Microsoft Visual C++ Redistributable：

- https://aka.ms/vs/17/release/vc_redist.x64.exe

安装后重启电脑或重新打开命令行。

---

## 10. 直接运行源码版本

在 Anaconda Prompt 中：

```bat
conda activate autometrology
cd /d C:\Projects\AutoMetrology
python main.py
```

也可以启动时直接加载 DXF：

```bat
python main.py C:\Data\xintai.dxf
```

预期结果：

- 弹出 `CAD Inspection Tool` 主窗口。
- 可以打开 DXF。
- 可以打开 Measurement Window。
- 如果安装了相机 SDK，可以在 Registration Panel 中刷新并打开相机。

---

## 11. 基本生产操作流程

1. 点击 `Open DXF`，加载生产图纸。
2. 打开 `View` → `Registration Panel`。
3. 选择或创建生产参数 Profile。
4. 打开相机，确认实时画面正常。
5. 设置 CAD P1 / P2 和 ROI P1 / P2。
6. 点击 `Auto Register`。
7. 打开 `Measurement Window`。
8. 加载或输入查询，例如：

```text
lines(794aad97, 23d434fc), 0.5565
arcs(3ff08234), 0.0200
circle(1A75E), 0.1000
```

9. 点击 `Evaluate` 做离线测量验证。
10. 点击 `Run Production` 执行采图、自动注册、测量和日志保存。
11. 点击 `View Logs` 查看生产测量日志。

---

## 12. PyInstaller 打包 EXE

仓库根目录提供 Windows 10 spec 文件：

```text
AutoMetrology-windows10.spec
```

> PyInstaller 不能可靠地从 Linux 交叉打包 Windows EXE。请在 Windows 10 目标机或 Windows 构建机上执行以下命令。

执行打包：

```bat
conda activate autometrology
cd /d C:\Projects\AutoMetrology
pyinstaller --clean --noconfirm AutoMetrology-windows10.spec
```

打包完成后输出目录：

```text
dist\AutoMetrology\
```

主程序：

```text
dist\AutoMetrology\AutoMetrology.exe
```

运行测试：

```bat
dist\AutoMetrology\AutoMetrology.exe
```

---

## 13. 打包时包含 MindVision DLL（可选）

推荐做法是在目标机器安装 MindVision SDK，并把 Runtime 目录加入 `PATH`。

如果希望把 `MVCAMSDK_X64.dll` 复制到 EXE 目录，可在打包前设置环境变量：

```bat
set MINDVISION_SDK_RUNTIME=C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64
pyinstaller --clean --noconfirm AutoMetrology-windows10.spec
```

spec 文件会检查：

```text
%MINDVISION_SDK_RUNTIME%\MVCAMSDK_X64.dll
```

如果文件存在，会复制到 `dist\AutoMetrology\` 根目录。

注意：

- 只复制 DLL 不一定等同完整 SDK 安装。
- 相机驱动、USB/GigE 驱动、注册表信息仍可能依赖官方安装程序。
- 生产机器建议完整安装 MindVision SDK。

---

## 14. 分发到其他 Windows 10 机器

将整个目录复制过去：

```text
dist\AutoMetrology\
```

目标机器需要：

1. 安装 VC++ 运行时。
2. 如需相机，安装 MindVision SDK。
3. 如需 DWG 导入，安装 ODA File Converter。
4. 双击 `AutoMetrology.exe`。

不要只复制单个 EXE。PyInstaller one-folder 模式依赖同目录下的 Qt、OpenCV、numpy/scipy DLL 和资源文件。

---

## 15. 创建桌面启动脚本（源码方式）

如果不打包 EXE，可在桌面创建 `AutoMetrology.bat`：

```bat
@echo off
call C:\Users\你的用户名\miniconda3\Scripts\activate.bat autometrology
cd /d C:\Projects\AutoMetrology
python main.py
pause
```

双击即可启动源码版本。

---

## 16. 创建桌面快捷方式（EXE 方式）

1. 进入：

```text
C:\Projects\AutoMetrology\dist\AutoMetrology
```

2. 右键 `AutoMetrology.exe`。
3. 选择“发送到” → “桌面快捷方式”。
4. 确认快捷方式的“起始位置”为：

```text
C:\Projects\AutoMetrology\dist\AutoMetrology
```

---

## 17. 常见问题

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| `ModuleNotFoundError: No module named 'PySide6'` | 未激活环境或依赖未安装 | `conda activate autometrology` 后重新 `pip install -r requirements-windows.txt` |
| `No module named 'cv2'` | OpenCV 未安装 | `pip install opencv-python` |
| `No module named 'ezdxf'` | DXF 解析依赖缺失 | `pip install ezdxf` |
| `scipy is required` | 注册/ICP 依赖缺失 | `pip install scipy` |
| `diplib` 安装失败 | pip wheel 不匹配或网络问题 | 先使用 OpenCV fallback 测试；生产环境建议换 Python 3.11 x64 后重装 |
| 打开相机失败 | SDK 未安装、PATH 未配置、相机被占用 | 安装 MindVision SDK，配置 Runtime PATH，关闭其他相机软件 |
| `MVCAMSDK_X64.dll` not found | SDK Runtime 不在 DLL 搜索路径 | 将 `Runtime\Win64_x64` 加入系统 PATH，或设置 `MINDVISION_SDK_RUNTIME` 后重新打包 |
| EXE 双击无反应 | 依赖 DLL 缺失但无控制台输出 | 用 `cmd` 进入 dist 目录执行 `AutoMetrology.exe` 查看错误，或临时把 spec 的 `console=True` 重打包 |
| DXF 能打开但 DWG 不能 | 未安装 ODA File Converter | 安装 ODA，并在软件中配置转换器路径 |
| 图像与 CAD 错位 | 自动注册参数或 P1/P2 ROI 错误 | 检查 CAD P1/P2、ROI P1/P2 是否一一对应，重新 Auto Register |
| 测量结果 `No Measurement` | 未加载图像、未注册、边缘不清晰 | 先确认图像层、自动注册和曝光，再重新 Evaluate |
| PyInstaller 打包很慢 | PySide6/OpenCV/scipy 包较大 | 正常；首次打包可能需要几分钟 |

---

## 18. 推荐交付清单

部署给产线机器时建议同时交付：

- `dist\AutoMetrology\` 整个目录
- 当前生产 DXF 文件
- 当前测量 query 文件
- 相机标定参数说明
- 自动注册 P1/P2 和 ROI 参数截图
- MindVision SDK 安装包
- VC++ 运行时安装包
- 本文档 `docs\Windows10部署指南.md`

---

## 19. 重新拉取更新并重新打包

当开发机推送新版本后，在 Windows 10 部署机执行：

```bat
conda activate autometrology
cd /d C:\Projects\AutoMetrology
git pull
pip install -r requirements-windows.txt
pyinstaller --clean --noconfirm AutoMetrology-windows10.spec
```

然后重新测试：

```bat
dist\AutoMetrology\AutoMetrology.exe
```
