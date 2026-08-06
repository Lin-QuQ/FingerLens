#!/usr/bin/env python3
"""Real-time two-hand fingertip filters inspired by editorial motion graphics."""

from __future__ import annotations

import argparse
import math
import platform
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TIP_IDS = (4, 8, 12, 16, 20)
FILTER_NAMES = {
    1: "NEGATIVE CHROME",
    2: "ANIME INK",
    3: "OIL IMPASTO",
    4: "NOIR FILM",
    5: "SOLAR PRINT",
    6: "GLITCH MOSAIC",
    7: "THERMAL RELIEF",
    8: "PENCIL SKETCH",
    9: "POP ART",
    10: "WATERCOLOR",
    11: "CYANOTYPE",
    12: "HALFTONE",
    13: "EMBOSSED METAL",
    14: "NEON EDGE",
    15: "DUOTONE CUTOUT",
    16: "PIXEL COMIC",
    17: "X-RAY",
    18: "SEPIA GRAIN",
    19: "POSTER RELIEF",
    20: "PRISM MIRROR",
    21: "VAPORWAVE",
    22: "HOLOGRAM",
    23: "ULTRAVIOLET POSTER",
    24: "LIQUID CHROME",
    25: "CORAL RISOGRAPH",
    26: "CMYK OFFSET",
    27: "NEWSPRINT",
    28: "SUMI INK",
    29: "AURORA",
    30: "SUNSET HEAT",
    31: "LAGOON GLASS",
    32: "JADE RELIEF",
    33: "RGB ECHO",
    34: "CRT DREAM",
    35: "DATA RIBBONS",
    36: "MATRIX PHOSPHOR",
    37: "GOLD LEAF",
    38: "ROSE GOLD",
    39: "PEARL SHIFT",
    40: "OBSIDIAN",
}
FILTER_SETS = {
    1: (1, 2, 3, 4),
    2: (5, 6, 7, 8),
    3: (9, 10, 11, 12),
    4: (13, 14, 15, 16),
    5: (17, 18, 19, 20),
    6: (21, 22, 23, 24),
    7: (25, 26, 27, 28),
    8: (29, 30, 31, 32),
    9: (33, 34, 35, 36),
    10: (37, 38, 39, 40),
}
ZONE_COLORS = (
    (255, 54, 181),   # pink / blue in BGR
    (255, 159, 31),
    (56, 255, 235),
    (91, 255, 80),
)
@dataclass
class SmoothLandmarks:
    alpha: float = 0.58
    values: MutableMapping[str, np.ndarray] = field(default_factory=dict)

    def update(self, side: str, points: np.ndarray) -> np.ndarray:
        previous = self.values.get(side)
        if previous is None or previous.shape != points.shape:
            smoothed = points.astype(np.float32)
        else:
            smoothed = previous * (1.0 - self.alpha) + points * self.alpha
        self.values[side] = smoothed
        return smoothed

    def forget_missing(self, visible: Sequence[str]) -> None:
        for side in list(self.values):
            if side not in visible:
                del self.values[side]


@dataclass
class ClapCycleSwitcher:
    """Trigger after two palms come together and then move apart."""

    stable_frames: int = 1
    release_frames: int = 2
    close_center_ratio: float = 2.15
    close_gap_ratio: float = 0.72
    open_center_ratio: float = 2.65
    open_gap_ratio: float = 1.05
    armed_state: bool = False
    close_streak: int = 0
    apart_streak: int = 0
    cooldown: int = 0
    missing_frames: int = 0

    def update(self, hands: Mapping[str, np.ndarray]) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
        if "Left" not in hands or "Right" not in hands:
            self.missing_frames += 1
            # At the instant the palms overlap, MediaPipe often merges them
            # into one detection. Preserve/arm a just-observed clap through it.
            if not self.armed_state and self.close_streak > 0 and self.cooldown == 0:
                self.armed_state = True
            self.apart_streak = 0
            return False

        self.missing_frames = 0
        left, right = hands["Left"], hands["Right"]
        palm_ids = (0, 1, 2, 5, 9, 13, 17)
        left_center = left[list(palm_ids)].mean(axis=0)
        right_center = right[list(palm_ids)].mean(axis=0)
        left_size = max(
            float(np.linalg.norm(left[5] - left[17])),
            float(np.linalg.norm(left[0] - left[9])),
        )
        right_size = max(
            float(np.linalg.norm(right[5] - right[17])),
            float(np.linalg.norm(right[0] - right[9])),
        )
        palm_size = max((left_size + right_size) * 0.5, 1.0)
        center_ratio = float(np.linalg.norm(left_center - right_center)) / palm_size
        left_palm = left[list(palm_ids)]
        right_palm = right[list(palm_ids)]
        pairwise = left_palm[:, None, :] - right_palm[None, :, :]
        gap_ratio = float(np.linalg.norm(pairwise, axis=2).min()) / palm_size

        if not self.armed_state:
            palms_near = (
                center_ratio <= self.close_center_ratio
                or gap_ratio <= self.close_gap_ratio
            )
            self.close_streak = self.close_streak + 1 if palms_near else 0
            if self.cooldown == 0 and self.close_streak >= self.stable_frames:
                self.armed_state = True
                self.apart_streak = 0
            return False

        palms_apart = (
            center_ratio >= self.open_center_ratio
            and gap_ratio >= self.open_gap_ratio
        )
        if palms_apart:
            self.apart_streak += 1
        else:
            self.apart_streak = 0
        if self.apart_streak >= self.release_frames:
            self.armed_state = False
            self.close_streak = 0
            self.apart_streak = 0
            self.cooldown = 15
            return True
        return False

    @property
    def armed(self) -> bool:
        return self.armed_state


def ensure_model(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"首次运行：正在下载 MediaPipe 手部模型到 {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "模型下载失败。请检查网络后重试，或手动下载：\n"
            f"{MODEL_URL}\n保存到：{path}"
        ) from exc
    return path


def make_landmarker(model_path: Path):
    # Explicit CPU inference is portable and fast enough for this model. It
    # also avoids relying on platform-specific GPU/OpenGL initialization.
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
    )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def normalize_hands(
    result,
    width: int,
    height: int,
    smoother: SmoothLandmarks,
    selfie_mirrored: bool = True,
):
    """Return anatomical Left/Right hands, repairing rare duplicate labels."""
    candidates = []
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if not categories:
            continue
        side = categories[0].category_name.title()
        # MediaPipe's handedness convention assumes a mirrored selfie image.
        if not selfie_mirrored and side in ("Left", "Right"):
            side = "Right" if side == "Left" else "Left"
        score = float(categories[0].score)
        points = np.array(
            [[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32
        )
        candidates.append((side, score, points))

    hands: Dict[str, np.ndarray] = {}
    for side, score, points in sorted(candidates, key=lambda item: item[1], reverse=True):
        if side in ("Left", "Right") and side not in hands:
            hands[side] = points

    # If the classifier assigns the same side twice, spatial order provides a
    # stable selfie-view fallback. Mirrored camera: user's right hand is left.
    if len(candidates) == 2 and len(hands) < 2:
        ordered = sorted((item[2] for item in candidates), key=lambda p: p[0, 0])
        hands = {"Right": ordered[0], "Left": ordered[1]}

    for side in list(hands):
        hands[side] = smoother.update(side, hands[side])
    smoother.forget_missing(hands.keys())
    return hands


def polygon_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points.astype(np.int32), 255, lineType=cv2.LINE_AA)
    return mask


def palette_map(gray: np.ndarray, colors: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Map luminance through a compact BGR palette with smooth transitions."""
    anchors = np.linspace(0, 255, len(colors), dtype=np.float32)
    values = np.arange(256, dtype=np.float32)
    lut = np.empty((256, 1, 3), dtype=np.uint8)
    for channel in range(3):
        channel_values = [color[channel] for color in colors]
        lut[:, 0, channel] = np.interp(values, anchors, channel_values).astype(np.uint8)
    return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)


def beauty_filter(frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Apply subtle skin-selective smoothing while preserving the background."""
    if strength <= 0.0:
        return frame

    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / max(width, 1))
    if scale < 1.0:
        working = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        working = frame

    ycrcb = cv2.cvtColor(working, cv2.COLOR_BGR2YCrCb)
    luminance, cr, cb = cv2.split(ycrcb)
    skin_mask = (
        (luminance > 38)
        & (cr > 130) & (cr < 182)
        & (cb > 72) & (cb < 138)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    smooth = cv2.bilateralFilter(
        working,
        7,
        38.0 + strength * 32.0,
        5.0 + strength * 4.0,
    )
    # Restore part of the original detail so skin stays natural rather than
    # looking like a uniform blur.
    smooth = cv2.addWeighted(smooth, 0.82, working, 0.18, 0)

    if scale < 1.0:
        smooth = cv2.resize(smooth, (width, height), interpolation=cv2.INTER_CUBIC)
        skin_mask = cv2.resize(
            skin_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )

    blended = cv2.addWeighted(frame, 1.0 - strength, smooth, strength, 0)
    result = frame.copy()
    cv2.copyTo(blended, skin_mask, result)
    return result


def fashion_filter(frame: np.ndarray, phase: float, style: int = 1) -> np.ndarray:
    """Create visibly different art treatments revealed by the finger zones."""
    height, width = frame.shape[:2]
    if style == 1:  # negative film + embossed chrome
        negative = 255 - frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        emboss = cv2.filter2D(gray, -1, np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]))
        emboss = cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
        filtered = cv2.addWeighted(negative, 0.78, emboss, 0.38, 8)
        shift = int(8 + 5 * math.sin(phase * 1.7))
        filtered[..., 0] = np.roll(filtered[..., 0], shift, axis=1)
        filtered[..., 2] = np.roll(filtered[..., 2], -shift, axis=1)
    elif style == 2:  # anime: flat paint bounded by black ink
        scale = min(1.0, 640.0 / width)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        paint = cv2.bilateralFilter(small, 9, 85, 85)
        paint = (paint // 42) * 42
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5
        )
        paint = cv2.bitwise_and(paint, paint, mask=ink)
        filtered = cv2.resize(paint, (width, height), interpolation=cv2.INTER_NEAREST)
    elif style == 3:  # oil painting / impasto
        scale = min(1.0, 520.0 / width)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "oilPainting"):
            painted = cv2.xphoto.oilPainting(small, 7, 1)
        else:
            painted = cv2.pyrMeanShiftFiltering(small, 18, 34)
        soft = cv2.GaussianBlur(painted, (0, 0), 1.1)
        detail = cv2.addWeighted(painted, 1.35, soft, -0.35, 0)
        filtered = cv2.resize(detail, (width, height), interpolation=cv2.INTER_CUBIC)
    elif style == 4:  # high-contrast monochrome film
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.7)
        grain_y, grain_x = np.indices(gray.shape)
        grain = ((grain_x * 17 + grain_y * 31 + int(phase * 90)) % 29 - 14).astype(np.int16)
        gray = np.clip(gray.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edges = cv2.Canny(gray, 48, 120)
        filtered[edges > 0] = (8, 8, 8)
    elif style == 5:  # solarized photographic print
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values = np.arange(256, dtype=np.int16)
        lut = np.where(values < 122, values * 2, (255 - values) * 2)
        solar = cv2.LUT(gray, np.clip(lut, 0, 255).astype(np.uint8))
        filtered = cv2.applyColorMap(solar, cv2.COLORMAP_TWILIGHT_SHIFTED)
        contours = cv2.Canny(gray, 42, 112)
        filtered[contours > 0] = 255 - filtered[contours > 0]
    elif style == 6:  # blocky datamosh / collage
        block = max(6, min(width, height) // 38)
        tiny = cv2.resize(frame, (max(2, width // block), max(2, height // block)), interpolation=cv2.INTER_AREA)
        filtered = cv2.resize(tiny, (width, height), interpolation=cv2.INTER_NEAREST)
        shift = int(14 + 12 * math.sin(phase))
        filtered[..., 0] = np.roll(filtered[..., 0], shift, axis=1)
        filtered[..., 2] = np.roll(filtered[..., 2], -shift, axis=1)
        band_height = max(8, height // 15)
        for band in range(0, height, band_height * 3):
            end = min(height, band + band_height)
            filtered[band:end] = np.roll(filtered[band:end], shift * 3, axis=1)
            if (band // band_height) % 2 == 0:
                filtered[band:end] = 255 - filtered[band:end]
    elif style == 7:  # thermal relief
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        filtered = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        relief = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        filtered[cv2.convertScaleAbs(relief) > 48] = (255, 255, 255)
    elif style == 8:  # graphite pencil
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverse = 255 - gray
        blur = cv2.GaussianBlur(inverse, (0, 0), 7)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        paper = np.clip(sketch.astype(np.int16) + 12, 0, 255).astype(np.uint8)
        filtered = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    elif style == 9:  # pop-art screen print
        smooth = cv2.bilateralFilter(frame, 7, 60, 60)
        filtered = (smooth // 64) * 64
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 28) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int16) * 2 + 45, 0, 255)
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 55, 130)
        filtered[edges > 0] = (0, 0, 0)
    elif style == 10:  # watercolor wash
        scale = min(1.0, 360.0 / max(width, 1))
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        wash = cv2.pyrMeanShiftFiltering(small, 13, 26)
        wash = cv2.bilateralFilter(wash, 7, 45, 45)
        filtered = cv2.resize(wash, (width, height), interpolation=cv2.INTER_CUBIC)
    elif style == 11:  # cyanotype print
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.8, (6, 6)).apply(gray)
        filtered = np.empty_like(frame)
        filtered[..., 0] = np.clip(gray.astype(np.float32) * 1.08 + 22, 0, 255)
        filtered[..., 1] = np.clip(gray.astype(np.float32) * 0.62, 0, 255)
        filtered[..., 2] = np.clip(gray.astype(np.float32) * 0.20, 0, 255)
    elif style == 12:  # graphic halftone
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bayer = np.array([[15, 7, 13, 5], [3, 11, 1, 9], [12, 4, 14, 6], [0, 8, 2, 10]], dtype=np.uint8) * 16
        threshold = np.tile(bayer, (math.ceil(height / 4), math.ceil(width / 4)))[:height, :width]
        dots = np.where(gray > threshold, 245, 18).astype(np.uint8)
        filtered = cv2.applyColorMap(dots, cv2.COLORMAP_BONE)
    elif style == 13:  # hammered / embossed metal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        relief = cv2.filter2D(gray, cv2.CV_16S, kernel)
        relief = cv2.convertScaleAbs(relief, alpha=1.7, beta=72)
        filtered = cv2.applyColorMap(relief, cv2.COLORMAP_BONE)
    elif style == 14:  # neon contour drawing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 42, 105)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        color_a = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
        color_b = cv2.applyColorMap(np.roll(edges, 3, axis=1), cv2.COLORMAP_HOT)
        filtered = cv2.addWeighted(color_a, 0.72, color_b, 0.68, 0)
        filtered[edges == 0] = (5, 2, 12)
    elif style == 15:  # hard duotone cutout
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        levels = (gray // 64).astype(np.uint8)
        palette = np.array(
            [[18, 8, 42], [170, 35, 110], [255, 105, 38], [255, 236, 92]],
            dtype=np.uint8,
        )
        filtered = palette[levels]
    elif style == 16:  # pixel comic
        block = max(4, min(width, height) // 30)
        tiny = cv2.resize(frame, (max(2, width // block), max(2, height // block)), interpolation=cv2.INTER_AREA)
        tiny = (tiny // 51) * 51
        filtered = cv2.resize(tiny, (width, height), interpolation=cv2.INTER_NEAREST)
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 70, 155)
        filtered[edges > 0] = 0
    elif style == 17:  # x-ray plate
        gray = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(3.4, (7, 7)).apply(gray)
        filtered = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
        edges = cv2.Canny(gray, 45, 110)
        filtered[edges > 0] = (255, 255, 225)
    elif style == 18:  # aged sepia grain
        transform = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]],
            dtype=np.float32,
        )
        filtered = cv2.transform(frame, transform)
        yy, xx = np.indices((height, width))
        grain = ((xx * 13 + yy * 29 + int(phase * 70)) % 23 - 11)[..., None]
        filtered = np.clip(filtered.astype(np.int16) + grain, 0, 255)
    elif style == 19:  # posterized relief
        blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
        sharp = cv2.addWeighted(frame, 1.7, blur, -0.7, 0)
        filtered = (sharp // 43) * 43
        relief = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_16S, ksize=3)
        filtered[cv2.convertScaleAbs(relief) > 44] = (18, 18, 18)
    elif style == 20:  # prismatic mirrored slices
        filtered = frame.copy()
        half = max(1, width // 2)
        mirrored = cv2.flip(frame[:, :half], 1)
        filtered[:, width - half:] = mirrored[:, :half]
        shift = int(5 + 5 * math.sin(phase))
        filtered[..., 0] = np.roll(filtered[..., 0], shift, axis=0)
        filtered[..., 2] = np.roll(filtered[..., 2], -shift, axis=1)
        stripe = max(6, height // 18)
        for y in range(0, height, stripe * 2):
            filtered[y:y + stripe] = cv2.flip(filtered[y:y + stripe], 1)
    elif style == 21:  # coordinated neon set: vaporwave gradient
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        filtered = palette_map(
            gray, ((28, 4, 38), (190, 30, 245), (255, 180, 35), (245, 250, 190)))
        edges = cv2.Canny(gray, 55, 135)
        filtered[edges > 0] = (255, 235, 90)
    elif style == 22:  # animated holographic foil
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices(gray.shape)
        hue = ((gray.astype(np.int16) // 2 + xx // 5 + yy // 8 + int(phase * 18)) % 180).astype(np.uint8)
        hsv = np.dstack((hue, np.full_like(gray, 205), cv2.equalizeHist(gray)))
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        filtered[::5] = np.clip(filtered[::5].astype(np.int16) + 38, 0, 255)
    elif style == 23:  # ultraviolet graphic poster
        smooth = cv2.bilateralFilter(frame, 7, 55, 55)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        gray = (gray // 51) * 51
        filtered = palette_map(
            gray, ((12, 2, 24), (95, 10, 185), (255, 35, 235), (255, 210, 55)))
        filtered[cv2.Canny(gray, 48, 118) > 0] = (5, 2, 12)
    elif style == 24:  # liquid chrome with neon reflections
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
        relief = cv2.Laplacian(smooth, cv2.CV_16S, ksize=3)
        chrome = cv2.convertScaleAbs(relief, alpha=2.4, beta=72)
        chrome = cv2.addWeighted(chrome, 0.72, cv2.equalizeHist(gray), 0.58, 0)
        filtered = palette_map(
            chrome, ((4, 2, 10), (110, 15, 150), (255, 120, 40), (255, 255, 255)))
    elif style == 25:  # warm coral risograph
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        levels = (cv2.GaussianBlur(gray, (3, 3), 0) // 64) * 64
        filtered = palette_map(
            levels, ((18, 20, 50), (45, 55, 160), (70, 110, 245), (225, 240, 250)))
        yy, xx = np.indices(gray.shape)
        dots = ((xx + yy) % 6 == 0) & (gray < 175)
        filtered[dots] = (20, 18, 45)
    elif style == 26:  # imperfect CMYK offset print
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ink = 255 - cv2.equalizeHist(gray)
        cyan = np.roll(ink, 3, axis=1)
        magenta = np.roll(ink, -3, axis=0)
        yellow = np.roll(ink, 2, axis=1)
        filtered = np.empty_like(frame)
        filtered[..., 0] = 255 - np.maximum(magenta // 2, yellow)
        filtered[..., 1] = 255 - np.maximum(cyan // 2, magenta)
        filtered[..., 2] = 255 - np.maximum(cyan, yellow // 2)
    elif style == 27:  # editorial newsprint
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        paper = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
        )
        yy, xx = np.indices(gray.shape)
        halftone = ((xx % 4 == 0) & (yy % 4 == 0) & (gray < 190))
        filtered = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
        filtered = np.where(filtered > 0, np.array((225, 238, 246), np.uint8), 18)
        filtered[halftone] = (35, 25, 70)
    elif style == 28:  # warm paper and sumi ink
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wash = cv2.bilateralFilter(gray, 9, 72, 72)
        ink = cv2.normalize(wash, None, 20, 245, cv2.NORM_MINMAX)
        filtered = palette_map(
            ink, ((8, 7, 12), (42, 38, 52), (165, 190, 215), (232, 244, 248)))
        contours = cv2.Canny(wash, 35, 92)
        filtered[contours > 0] = (5, 5, 8)
    elif style == 29:  # aurora palette
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices(gray.shape)
        waves = (24 * np.sin(xx / 24.0 + phase) + 18 * np.cos(yy / 19.0 - phase)).astype(np.int16)
        glow = np.clip(gray.astype(np.int16) + waves, 0, 255).astype(np.uint8)
        filtered = palette_map(
            glow, ((20, 8, 35), (125, 35, 90), (120, 245, 40), (255, 245, 175)))
    elif style == 30:  # sunset heat map
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        solar = np.where(gray < 150, gray, 255 - gray // 2).astype(np.uint8)
        filtered = palette_map(
            solar, ((35, 5, 45), (90, 20, 190), (35, 95, 255), (100, 245, 255)))
        filtered[cv2.Canny(gray, 60, 145) > 0] = (60, 15, 110)
    elif style == 31:  # translucent lagoon glass
        small_scale = min(1.0, 600.0 / max(width, 1))
        small = cv2.resize(frame, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA)
        glass = cv2.pyrMeanShiftFiltering(small, 12, 25)
        glass = cv2.resize(glass, (width, height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(glass, cv2.COLOR_BGR2GRAY)
        filtered = palette_map(
            gray, ((35, 18, 8), (135, 105, 10), (220, 220, 70), (255, 252, 205)))
        filtered[cv2.Canny(gray, 42, 105) > 0] = (245, 255, 225)
    elif style == 32:  # carved jade relief
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        carved = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_16S, kernel), alpha=1.6, beta=65)
        filtered = palette_map(
            carved, ((12, 28, 18), (45, 105, 35), (115, 190, 85), (210, 245, 220)))
    elif style == 33:  # strong RGB temporal-style echo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shift = int(7 + 6 * math.sin(phase * 1.3))
        filtered = np.dstack((
            np.roll(gray, shift, axis=1), gray, np.roll(gray, -shift, axis=1)
        ))
        filtered = cv2.convertScaleAbs(filtered, alpha=1.28, beta=-18)
    elif style == 34:  # dreamy CRT phosphor
        soft = cv2.GaussianBlur(frame, (0, 0), 1.3)
        hsv = cv2.cvtColor(soft, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 24) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int16) * 1.5 + 45, 0, 255)
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        filtered[1::3] = (filtered[1::3].astype(np.float32) * 0.52).astype(np.uint8)
    elif style == 35:  # coordinated digital data ribbons
        filtered = frame.copy()
        band = max(4, height // 18)
        for index, y in enumerate(range(0, height, band)):
            offset = int((index % 5 - 2) * 7 + 9 * math.sin(phase + index))
            filtered[y:y + band] = np.roll(filtered[y:y + band], offset, axis=1)
        filtered[..., 0] = np.roll(filtered[..., 0], 5, axis=1)
        filtered[..., 2] = np.roll(filtered[..., 2], -5, axis=1)
        filtered = (filtered // 32) * 32
    elif style == 36:  # matrix green phosphor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        filtered = palette_map(
            gray, ((0, 8, 0), (4, 55, 8), (35, 210, 45), (190, 255, 205)))
        yy, xx = np.indices(gray.shape)
        grid = ((xx % 7 == 0) | (yy % 7 == 0)) & (gray < 145)
        filtered[grid] = (0, 28, 0)
    elif style == 37:  # hammered gold leaf
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        relief = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_16S, ksize=3)
        metal = cv2.addWeighted(gray, 0.75, cv2.convertScaleAbs(relief, alpha=2.0, beta=45), 0.65, 0)
        filtered = palette_map(
            metal, ((3, 8, 15), (12, 55, 105), (35, 155, 225), (205, 245, 255)))
    elif style == 38:  # satin rose gold
        gray = cv2.cvtColor(cv2.bilateralFilter(frame, 9, 65, 65), cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices(gray.shape)
        sheen = (22 * np.sin((xx + yy) / 34.0 + phase)).astype(np.int16)
        satin = np.clip(gray.astype(np.int16) + sheen, 0, 255).astype(np.uint8)
        filtered = palette_map(
            satin, ((18, 8, 30), (65, 45, 115), (135, 145, 225), (230, 238, 255)))
    elif style == 39:  # iridescent pearl shift
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yy, xx = np.indices(gray.shape)
        hue = ((gray.astype(np.int16) // 6 + xx // 14 - yy // 18 + int(phase * 8)) % 180).astype(np.uint8)
        saturation = np.clip(115 - gray.astype(np.int16) // 4, 38, 115).astype(np.uint8)
        value = np.clip(gray.astype(np.int16) + 65, 0, 255).astype(np.uint8)
        filtered = cv2.cvtColor(np.dstack((hue, saturation, value)), cv2.COLOR_HSV2BGR)
    else:  # obsidian with restrained violet-gold edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark = cv2.convertScaleAbs(gray, alpha=0.58, beta=-18)
        filtered = palette_map(
            dark, ((2, 1, 5), (18, 6, 28), (62, 25, 90), (130, 100, 175)))
        edges = cv2.Canny(gray, 38, 96)
        gold = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        filtered[gold > 0] = (45, 185, 245)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def glow_polyline(layer: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    glow = np.zeros_like(layer)
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(glow, [pts], True, color, 6, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 5)
    cv2.addWeighted(glow, 0.65, layer, 1.0, 0, layer)
    cv2.polylines(layer, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(layer, [pts], True, color, 1, cv2.LINE_AA)


def draw_zones(
    frame: np.ndarray,
    hands: Mapping[str, np.ndarray],
    phase: float,
    style: int,
) -> np.ndarray:
    if "Left" not in hands or "Right" not in hands:
        return frame

    left, right = hands["Left"], hands["Right"]
    overlay = frame.copy()
    filter_ids = FILTER_SETS[style]
    frame_height, frame_width = frame.shape[:2]

    for zone, (a, b) in enumerate(zip(TIP_IDS[:-1], TIP_IDS[1:])):
        quad = np.array([left[a], left[b], right[b], right[a]], dtype=np.float32)
        # Sort vertices around their centroid so crossed hands still form a region.
        center = quad.mean(axis=0)
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        quad = quad[np.argsort(angles)]
        x, y, box_width, box_height = cv2.boundingRect(quad.astype(np.int32))
        padding = 4
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(frame_width, x + box_width + padding)
        y1 = min(frame_height, y + box_height + padding)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue

        source_roi = frame[y0:y1, x0:x1]
        filtered_roi = fashion_filter(source_roi, phase + zone * 0.37, filter_ids[zone])
        local_quad = quad - np.array([x0, y0], dtype=np.float32)
        mask = polygon_mask(source_roi.shape[:2], local_quad)
        color = ZONE_COLORS[(zone + style - 1) % len(ZONE_COLORS)]
        alpha = (mask.astype(np.float32) / 255.0 * 0.94)[..., None]
        target_roi = overlay[y0:y1, x0:x1]
        target_roi[:] = (
            target_roi.astype(np.float32) * (1.0 - alpha)
            + filtered_roi.astype(np.float32) * alpha
        ).astype(np.uint8)
        glow_polyline(overlay, quad, color)
    return overlay


def draw_interface(
    frame: np.ndarray,
    fps: float,
    style: int,
    hand_count: int,
    clap_armed: bool,
    beauty_enabled: bool,
) -> None:
    height, width = frame.shape[:2]
    panel_right = min(width - 16, 455)
    cv2.rectangle(frame, (16, 16), (panel_right, 82), (8, 8, 12), -1)
    cv2.line(frame, (16, 16), (panel_right, 16), ZONE_COLORS[(style - 1) % 4], 3)
    cv2.putText(
        frame, "Lin-menmen", (28, 45),
        cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"ART SET {style:02d} / {len(FILTER_SETS):02d}", (182, 45),
        cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA,
    )
    gesture_text = "RELEASE PALMS" if clap_armed else "CLAP PALMS TO CHANGE"
    cv2.putText(
        frame, f"{gesture_text}   {fps:04.1f} FPS   {hand_count}/2",
        (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (83, 238, 255), 1, cv2.LINE_AA,
    )
    beauty_state = "ON" if beauty_enabled else "OFF"
    cv2.putText(
        frame, f"[1-9/0] SET   [B] BEAUTY {beauty_state}   [H] HUD   [M] MIRROR   [Q] QUIT",
        (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )
    for corner_x in (12, width - 12):
        direction = 1 if corner_x < width // 2 else -1
        cv2.line(frame, (corner_x, 110), (corner_x + direction * 26, 110), (255, 255, 255), 1)
        cv2.line(frame, (corner_x, 110), (corner_x, 136), (255, 255, 255), 1)


def draw_brand(frame: np.ndarray, style: int) -> None:
    """Keep the creator mark visible even when the optional HUD is hidden."""
    cv2.rectangle(frame, (16, 16), (166, 55), (8, 8, 12), -1)
    cv2.line(frame, (16, 16), (166, 16), ZONE_COLORS[(style - 1) % 4], 3)
    cv2.putText(
        frame, "Lin-menmen", (28, 44),
        cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA,
    )


def camera_frame_is_black(frame: np.ndarray | None) -> bool:
    """Detect all/near-black frames returned when camera access fails."""
    if frame is None or frame.size == 0:
        return True
    sample = frame[::8, ::8]
    return float(sample.max()) < 12.0 and float(sample.mean()) < 3.0


def camera_backend_candidates(
    system_name: str | None = None,
    requested: str = "auto",
) -> list[tuple[str, int]]:
    """Return ordered OpenCV camera backends for the current platform."""
    backends = {
        "any": cv2.CAP_ANY,
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    }
    if requested != "auto":
        return [(requested, backends[requested])]

    current = system_name or platform.system()
    if current == "Darwin":
        names = ("avfoundation", "any")
    elif current == "Windows":
        # DirectShow is usually the least troublesome for USB webcams; MSMF is
        # retained as a fallback for integrated and virtual cameras.
        names = ("dshow", "msmf", "any")
    elif current == "Linux":
        names = ("v4l2", "any")
    else:
        names = ("any",)
    return [(name, backends[name]) for name in names]


def camera_help(system_name: str | None = None) -> str:
    current = system_name or platform.system()
    if current == "Darwin":
        return (
            "系统设置 → 隐私与安全性 → 摄像头：允许当前终端或 IDE；"
            "修改权限后请完全退出并重新打开该应用。"
        )
    if current == "Windows":
        return (
            "Windows 设置 → 隐私和安全性 → 摄像头：开启“摄像头访问”和"
            "“允许桌面应用访问摄像头”；并关闭可能占用摄像头的会议/直播软件。"
        )
    return (
        "请确认当前用户有权访问 /dev/video*，并关闭可能占用摄像头的应用。"
    )


def open_camera(
    index: int,
    width: int,
    height: int,
    requested_backend: str = "auto",
) -> cv2.VideoCapture:
    """Open a camera with platform-specific backends and automatic fallback."""
    attempted = []
    opened_but_black = []
    for backend_name, backend in camera_backend_candidates(requested=requested_backend):
        attempted.append(backend_name)
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        # Start at the camera's native resolution. Some devices return black
        # frames when an unsupported size is forced during initialization.
        valid_frame = False
        for _ in range(45):
            ok, warmup = capture.read()
            if ok and not camera_frame_is_black(warmup):
                valid_frame = True
                break
            time.sleep(0.025)

        if not valid_frame:
            opened_but_black.append(backend_name)
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"摄像头 {index} 已连接，后端：{backend_name}；"
            f"请求 {width}x{height}，实际 {actual_width}x{actual_height}"
        )
        return capture

    detail = (
        "摄像头可以打开但持续返回黑画面。"
        if opened_but_black
        else "摄像头无法打开。"
    )
    raise RuntimeError(
        f"{detail} 编号：{index}；已尝试后端：{', '.join(attempted)}。\n"
        f"{camera_help()}\n"
        "若使用外接或虚拟摄像头，请尝试 --camera 1；也可用 --backend 指定后端。"
    )


def run(args: argparse.Namespace) -> None:
    model_path = ensure_model(args.model)
    capture = open_camera(args.camera, args.width, args.height, args.backend)
    smoother = SmoothLandmarks(args.smoothing)
    clap_switcher = ClapCycleSwitcher()
    mirror, show_hud, style = not args.no_mirror, True, args.style
    beauty_strength = args.beauty if args.beauty > 0.0 else 0.35
    beauty_enabled = args.beauty > 0.0
    style_keys = {ord(str(number)): number for number in range(1, 10)}
    style_keys[ord("0")] = 10
    switch_flash = 0
    started = time.perf_counter()
    last_tick, smooth_fps = started, 0.0
    frame_index = 0
    window = "FingerLens — Q to quit"

    try:
        with make_landmarker(model_path) as landmarker:
            # WINDOW_AUTOSIZE avoids a Cocoa/OpenCV issue where a resizable
            # window can vanish while crossing macOS displays with different
            # Retina scale factors. The image stays at its native resolution.
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("摄像头读帧失败。请关闭其他占用摄像头的应用后重试。")
                if camera_frame_is_black(frame):
                    raise RuntimeError(
                        "摄像头在运行中返回黑画面。"
                        f"{camera_help()}"
                    )
                if mirror:
                    frame = cv2.flip(frame, 1)
                if beauty_enabled:
                    frame = beauty_filter(frame, beauty_strength)
                height, width = frame.shape[:2]
                detection_scale = min(1.0, args.detect_width / max(width, 1))
                if detection_scale < 1.0:
                    detection_frame = cv2.resize(
                        frame, None,
                        fx=detection_scale, fy=detection_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    detection_frame = frame
                rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.perf_counter() - started) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = normalize_hands(
                    result, width, height, smoother, selfie_mirrored=mirror
                )
                if clap_switcher.update(hands):
                    style = style % len(FILTER_SETS) + 1
                    switch_flash = 7

                phase = time.perf_counter() * 2.2
                output = draw_zones(frame, hands, phase, style)
                if switch_flash > 0:
                    flash_alpha = switch_flash / 18.0
                    output = cv2.addWeighted(
                        output, 1.0 - flash_alpha,
                        np.full_like(output, (255, 255, 255)), flash_alpha, 0,
                    )
                    switch_flash -= 1

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_tick, 1e-6)
                smooth_fps = instant_fps if frame_index == 0 else smooth_fps * 0.9 + instant_fps * 0.1
                last_tick, frame_index = now, frame_index + 1
                if show_hud:
                    draw_interface(
                        output, smooth_fps, style, len(hands),
                        clap_switcher.armed, beauty_enabled,
                    )
                else:
                    draw_brand(output, style)

                cv2.imshow(window, output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in style_keys:
                    style = style_keys[key]
                elif key == ord("h"):
                    show_hud = not show_hud
                elif key == ord("m"):
                    mirror = not mirror
                    smoother.values.clear()
                elif key == ord("b"):
                    beauty_enabled = not beauty_enabled
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时双手指尖区域滤镜")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument(
        "--backend",
        choices=("auto", "avfoundation", "dshow", "msmf", "v4l2", "any"),
        default="auto",
        help="摄像头后端；默认按操作系统自动选择并回退",
    )
    parser.add_argument("--width", type=int, default=1920, help="请求的采集宽度，默认 1920")
    parser.add_argument("--height", type=int, default=1080, help="请求的采集高度，默认 1080")
    parser.add_argument(
        "--detect-width", type=int, default=960,
        help="手部识别使用的最大宽度，默认 960；不影响输出清晰度",
    )
    parser.add_argument("--style", type=int, choices=tuple(FILTER_SETS), default=1)
    parser.add_argument(
        "--beauty", type=float, default=0.0,
        help="磨皮强度 0-1，默认关闭；按 B 以 0.35 开启",
    )
    parser.add_argument("--smoothing", type=float, default=0.58, help="0-1，越大越跟手")
    parser.add_argument("--no-mirror", action="store_true", help="关闭自拍镜像")
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "hand_landmarker.task",
    )
    args = parser.parse_args()
    if not 0.0 < args.smoothing <= 1.0:
        parser.error("--smoothing 必须在 (0, 1] 范围内")
    if not 0.0 <= args.beauty <= 1.0:
        parser.error("--beauty 必须在 [0, 1] 范围内")
    dimensions = (args.width, args.height, args.detect_width)
    if any(value <= 0 for value in dimensions):
        parser.error("采集和识别尺寸必须大于 0")
    return args


if __name__ == "__main__":
    run(parse_args())
