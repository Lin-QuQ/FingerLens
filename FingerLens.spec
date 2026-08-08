# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
)


ROOT = Path(SPEC).resolve().parent
IS_MACOS = sys.platform == "darwin"

# Do not collect every model shipped for MediaPipe Solutions. FingerLens uses
# only the Hand Landmarker task model bundled below; face, pose, iris, and
# segmentation models add tens of megabytes and are never loaded.
mediapipe_binaries = collect_dynamic_libs("mediapipe")

a = Analysis(
    [str(ROOT / "finger_lens.py")],
    pathex=[str(ROOT)],
    binaries=mediapipe_binaries,
    datas=[
        (str(ROOT / "models" / "hand_landmarker.task"), "models"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

icon_path = ROOT / "assets" / (
    "fingerlens.icns" if IS_MACOS else "fingerlens.ico"
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FingerLens",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FingerLens",
)

if IS_MACOS:
    app = BUNDLE(
        coll,
        name="FingerLens.app",
        icon=str(icon_path),
        bundle_identifier="com.linmenmen.fingerlens",
        version="1.0.0",
        info_plist={
            "CFBundleDisplayName": "FingerLens",
            "CFBundleName": "FingerLens",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1",
            "LSMinimumSystemVersion": "13.0",
            "NSCameraUsageDescription": "FingerLens 需要使用摄像头实时识别手势并生成艺术滤镜。画面仅在本机处理。",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
