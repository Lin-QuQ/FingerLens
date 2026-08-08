# FingerLens 打包与发布

普通用户不需要阅读本文件。源码开发者仍然使用 `requirements.txt`；只有制作免安装版时才需要下面的构建依赖。

## 自动构建（推荐）

`.github/workflows/build-release.yml` 提供两个独立构建环境：

- Windows x64：`windows-latest`
- macOS Apple Silicon：`macos-15`

在 GitHub 仓库的 **Actions → Build desktop apps → Run workflow** 可以手动构建测试包。构建成功后，从该次运行页面的 Artifacts 区域下载。

正式发布时创建并推送版本标签：

```bash
git tag v1.0.0
git push origin v1.0.0
```

工作流会自动执行：

1. 安装 Python 3.11 和固定版本依赖。
2. 运行全部单元测试。
3. 使用 `FingerLens.spec` 构建无终端窗口的桌面应用。
4. 校验最终可执行文件确实为 Windows x64 或 macOS arm64，防止错误架构混入发布包。
5. 使用打包后的程序加载 MediaPipe 模型并执行自检。
6. 生成两个 ZIP 文件。
7. 创建 GitHub Release 并上传 ZIP。

GitHub 会自动在 Release 中附带源码 ZIP 和 TAR.GZ。若需服务国内用户，可将两个免安装 ZIP 再上传到对应的 Gitee Release。

## 本机构建

PyInstaller 不是交叉编译器：Windows 包必须在 Windows 构建，macOS 包必须在 macOS 构建。

macOS Apple Silicon 使用统一的 arm64 指令集，一个安装包覆盖 M1、M2、M3、M4、M5 及后续 Apple 芯片。项目不再发布 Intel Mac 免安装版。

固定版本 OpenCV 的 macOS wheel 最低平台标签为 macOS 13，因此当前 macOS 免安装版和源码依赖均以 macOS 13 为最低版本。

```bash
python -m pip install -r packaging/requirements-build.txt
python -m unittest -v
pyinstaller --noconfirm --clean FingerLens.spec
```

产物位置：

- Windows：`dist/FingerLens/FingerLens.exe`
- macOS：`dist/FingerLens.app`

模型自检：

Windows PowerShell：

```powershell
.\dist\FingerLens\FingerLens.exe --self-test
```

macOS：

```bash
dist/FingerLens.app/Contents/MacOS/FingerLens --self-test
```

## 应用资源

- 原始图标：`assets/fingerlens-icon.png`
- Windows 图标：`assets/fingerlens.ico`
- macOS 图标：`assets/fingerlens.icns`
- 小白说明：`packaging/使用说明.txt`
- PyInstaller 配置：`FingerLens.spec`

macOS 包的 `Info.plist` 已包含摄像头用途说明和 Retina 支持。当前自动构建执行 ad-hoc 签名，但没有 Apple 公证或 Windows 商业代码签名，因此首次启动仍可能出现系统安全提醒。
