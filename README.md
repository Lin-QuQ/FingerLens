# FingerLens

<p align="center">
  <img src="assets/fingerlens-icon.png" width="144" alt="FingerLens 图标">
</p>

基于 Python、MediaPipe 和 OpenCV 的实时双手指尖艺术滤镜。程序识别左右手指尖，把两手相邻手指围成的四边形变成独立的动态滤镜区域；摄像头画面全程在本机处理。

## 选择你的版本

### 只想直接玩（无需 Python）

前往 [GitHub Releases](https://github.com/Lin-QuQ/FingerLens/releases/latest) 下载与你电脑对应的免安装包：

| 电脑 | 下载文件 | 启动方式 |
| --- | --- | --- |
| Windows 10/11 64 位 | `FingerLens-Windows-x64.zip` | 完整解压后双击 `FingerLens.exe` |
| Apple Silicon Mac（M1/M2/M3/M4/M5 及后续 Apple 芯片） | `FingerLens-macOS-AppleSilicon.zip` | 解压后双击 `FingerLens.app` |

免安装版已经包含 Python、MediaPipe、OpenCV 和手部模型。第一次启动需要允许摄像头权限。当前发布包未购买商业代码签名：Windows 如果出现 SmartScreen，请选择“更多信息”→“仍要运行”；macOS 如果提示无法验证开发者，请右键应用选择“打开”。压缩包内附有 `使用说明.txt`。

国内用户可以前往 [Gitee Releases](https://gitee.com/Lin-QuQ/FingerLens/releases) 下载相同的 Windows x64 和 macOS Apple Silicon 免安装包；[Gitee 仓库主页](https://gitee.com/Lin-QuQ/FingerLens) 同步提供完整源码。

### 想学习或修改代码

继续阅读下面的源码安装方法。仓库完整保留 Python 源码，并采用 MIT License。

## 功能

- 实时追踪两只手和十个指尖
- 四个指缝区域同时使用不同滤镜
- 10 套组合、共 40 种艺术滤镜
- 轻度肤色区域磨皮，可调强度并实时开关
- 双手像鼓掌一样合拢再分开，即可切换下一套
- 支持快捷键切换、镜像和隐藏界面
- 默认请求 1920×1080 高清画面，并用较小图像进行手部识别以兼顾帧率
- 自动适配 macOS、Windows 和 Linux 摄像头后端

## 系统要求

| 系统 | 支持情况 | 默认摄像头后端 |
| --- | --- | --- |
| macOS 13+（Apple Silicon） | 支持 | AVFoundation |
| Windows 10/11 x64 | 支持 | DirectShow → MSMF |
| Linux x86-64 | 实验性支持 | V4L2 |

源码运行需要 Python 3.11 和可用摄像头。Intel Mac 和 Windows ARM 暂不提供免安装版。

## 源码版安装

### 使用 Conda（推荐）

下载或克隆仓库后，在项目目录中运行：

```bash
conda env create -f environment.yml
conda activate fingerlens
python finger_lens.py
```

如果 `fingerlens` 环境已经存在，可以更新它：

```bash
conda env update -f environment.yml --prune
```

### 使用普通 Python

```bash
python -m venv .venv
```

macOS/Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python finger_lens.py
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python finger_lens.py
```

MediaPipe 手部模型已经包含在 `models/hand_landmarker.task` 中，正常安装和运行不需要访问 Google，也不需要代理。如果模型被误删，程序仍会尝试从 Google 官方地址自动下载作为备用方案；此时中国大陆网络可能需要代理。

## 使用方式

让两手掌心朝向镜头、五指分开并同时进入画面。只有检测到两只手时，四个跨手区域才会出现。画面不会绘制手部骨架、指节圆点或手指名称。

每一套包含 4 种同时出现的滤镜，依次作用于：

1. 拇指—食指
2. 食指—中指
3. 中指—无名指
4. 无名指—小指

双手像鼓掌一样合到一起，再向左右明显分开，即可切换下一套。靠近成功后，左上角会显示 `RELEASE PALMS`。

## 滤镜组合

| 套装 | 四个区域的滤镜 |
| --- | --- |
| 1 | 反片铬印 / 动漫墨线 / 厚涂油画 / 黑白胶片 |
| 2 | 日晒反转 / 故障拼贴 / 热成像浮雕 / 铅笔素描 |
| 3 | 波普印刷 / 水彩 / 蓝晒 / 半调网点 |
| 4 | 金属浮雕 / 霓虹轮廓 / 双色剪纸 / 像素漫画 |
| 5 | X 光 / 棕褐胶片 / 海报浮雕 / 棱镜镜像 |
| 6 | 蒸汽波 / 全息镭射 / 紫外海报 / 液态铬 |
| 7 | 珊瑚孔版印刷 / CMYK 套色 / 报纸网印 / 水墨 |
| 8 | 极光 / 落日热感 / 潟湖玻璃 / 翡翠浮雕 |
| 9 | RGB 残影 / CRT 梦境 / 数据缎带 / 矩阵荧光 |
| 10 | 金箔 / 玫瑰金 / 珍珠偏光 / 黑曜石 |

## 快捷键

- `1`–`9`：手动选择第 1–9 套滤镜
- `0`：手动选择第 10 套滤镜
- `B`：开启或关闭磨皮
- `H`：显示/隐藏界面信息
- `M`：切换自拍镜像
- `Q` 或 `Esc`：退出

## 命令行选项

```bash
python finger_lens.py --help
python finger_lens.py --camera 1
python finger_lens.py --width 1280 --height 720
python finger_lens.py --width 1920 --height 1080 --detect-width 1280
python finger_lens.py --beauty 0.5
python finger_lens.py --backend dshow
```

支持的摄像头后端：`auto`、`avfoundation`、`dshow`、`msmf`、`v4l2`、`any`。通常保持 `auto` 即可。

默认会向摄像头请求 1920×1080。程序启动后会打印“请求分辨率”和“实际分辨率”；实际值由摄像头硬件及驱动决定。`--detect-width` 只控制 MediaPipe 的识别开销，不会降低最终画面的分辨率。性能充足时可以设为 `1280`，追求更高帧率时可以设为 `720`。

磨皮默认关闭，运行时按 `B` 会以自然强度 `0.35` 开启，再按一次即可关闭。磨皮只在检测到的肤色区域进行柔化，尽量保留背景和轮廓细节。也可以使用 `--beauty 0.5` 指定强度并在启动时直接开启。该功能只使用 NumPy 和 OpenCV，在 macOS、Windows 和 Linux 上采用相同实现。

## 摄像头问题

### macOS

打开“系统设置 → 隐私与安全性 → 摄像头”，允许当前终端或 IDE。修改后完全退出并重新打开终端/IDE。

### Windows

打开“设置 → 隐私和安全性 → 摄像头”，同时开启：

- 摄像头访问
- 允许应用访问摄像头
- 允许桌面应用访问摄像头

如果默认模式失败，可以依次尝试：

```powershell
python finger_lens.py --backend dshow
python finger_lens.py --backend msmf
python finger_lens.py --camera 1
```

### 通用排查

- 关闭 FaceTime、微信、Teams、Zoom、OBS 等可能占用摄像头的应用
- 外接摄像头或虚拟摄像头通常使用 `--camera 1`
- 性能不足时使用 `--width 1280 --height 720 --detect-width 720`
- `--smoothing` 越小越稳定，越大越跟手

## 测试

```bash
python -m unittest -v
```

自动化测试覆盖滤镜输出、区域蒙版、关键点平滑、鼓掌状态机、黑帧识别和跨平台摄像头后端选择。由于摄像头驱动与硬件有关，发布新版本前仍建议分别在 macOS 和 Windows 真机测试。

## 打包与发布

仓库包含 PyInstaller 配置和 GitHub Actions 工作流。推送 `v*` 标签后，会分别在 Windows x64 和 macOS Apple Silicon 环境中校验架构、运行测试、构建免安装包、执行模型自检并创建 GitHub Release。维护者操作说明见 [PACKAGING.md](PACKAGING.md)。

## 隐私

摄像头帧只在本机内存中处理，不会上传。仓库已包含运行所需的手部模型，因此正常运行不需要联网；只有模型文件缺失时，程序才会尝试从 Google MediaPipe 官方地址重新下载。

## 第三方组件

手部识别使用 Google MediaPipe Hand Landmarker 模型。模型来源、校验值和许可证信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 开源协议

[MIT License](LICENSE)
