import unittest
from unittest.mock import patch

import numpy as np

from finger_lens import (
    ClapCycleSwitcher,
    SmoothLandmarks,
    FILTER_NAMES,
    FILTER_SETS,
    beauty_filter,
    camera_backend_candidates,
    camera_frame_is_black,
    camera_help,
    camera_index_candidates,
    draw_zones,
    fashion_filter,
    open_camera,
    polygon_mask,
    parse_args,
    window_is_closed,
)


class EffectTests(unittest.TestCase):
    def test_polygon_mask_fills_center(self):
        points = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        mask = polygon_mask((64, 64), points)
        self.assertEqual(mask[30, 30], 255)
        self.assertEqual(mask[2, 2], 0)

    def test_smoothing(self):
        smoother = SmoothLandmarks(alpha=0.5)
        first = np.zeros((21, 2), dtype=np.float32)
        second = np.full((21, 2), 10, dtype=np.float32)
        smoother.update("Left", first)
        result = smoother.update("Left", second)
        np.testing.assert_allclose(result, 5.0)

    def test_all_filter_styles_keep_shape(self):
        frame = np.full((96, 128, 3), 90, dtype=np.uint8)
        for style in FILTER_NAMES:
            result = fashion_filter(frame, 1.2, style)
            self.assertEqual(result.shape, frame.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_beauty_filter_preserves_shape_and_background(self):
        frame = np.full((96, 128, 3), (70, 110, 170), dtype=np.uint8)
        frame[24:72, 40:88] = (95, 145, 205)
        frame[36:60:2, 48:80:2] = (70, 105, 160)
        result = beauty_filter(frame, 0.5)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(beauty_filter(frame, 0.0), frame)

    def test_beauty_is_disabled_by_default(self):
        with patch("sys.argv", ["finger_lens.py"]):
            self.assertEqual(parse_args().beauty, 0.0)

    def test_ten_sets_cover_forty_unique_filters(self):
        self.assertEqual(len(FILTER_SETS), 10)
        filter_ids = [filter_id for group in FILTER_SETS.values() for filter_id in group]
        self.assertEqual(len(filter_ids), 40)
        self.assertEqual(len(set(filter_ids)), 40)
        self.assertEqual(set(filter_ids), set(FILTER_NAMES))

    def test_two_hands_change_zone_pixels(self):
        frame = np.full((160, 220, 3), 80, dtype=np.uint8)
        left = np.tile(np.array([30.0, 80.0]), (21, 1))
        right = np.tile(np.array([190.0, 80.0]), (21, 1))
        for i, tip in enumerate((4, 8, 12, 16, 20)):
            left[tip] = (30 + i * 8, 30 + i * 24)
            right[tip] = (190 - i * 8, 30 + i * 24)
        for style in FILTER_SETS:
            result = draw_zones(frame, {"Left": left, "Right": right}, 1.0, style)
            self.assertEqual(result.shape, frame.shape)
            self.assertGreater(
                np.mean(np.abs(result.astype(int) - frame.astype(int))), 1.0
            )

    def test_black_camera_frame_detection(self):
        self.assertTrue(camera_frame_is_black(np.zeros((64, 64, 3), dtype=np.uint8)))
        visible = np.zeros((64, 64, 3), dtype=np.uint8)
        visible[::8, ::8] = 80
        self.assertFalse(camera_frame_is_black(visible))

    def test_platform_camera_backends(self):
        self.assertEqual(
            [name for name, _ in camera_backend_candidates("Darwin")],
            ["avfoundation", "any"],
        )
        self.assertEqual(
            [name for name, _ in camera_backend_candidates("Windows")],
            ["dshow", "msmf", "any"],
        )
        self.assertEqual(
            [name for name, _ in camera_backend_candidates("Linux")],
            ["v4l2", "any"],
        )
        self.assertEqual(
            [name for name, _ in camera_backend_candidates("Windows", "msmf")],
            ["msmf"],
        )

    def test_camera_index_auto_scan_and_explicit_selection(self):
        self.assertEqual(camera_index_candidates(-1), (0, 1, 2))
        self.assertEqual(camera_index_candidates(1), (1,))
        with patch("sys.argv", ["finger_lens.py"]):
            self.assertEqual(parse_args().camera, -1)

    def test_auto_camera_scan_falls_back_from_zero_to_one(self):
        opened_indices = []

        class FakeCapture:
            def __init__(self, index, _backend):
                self.index = index
                opened_indices.append(index)

            def isOpened(self):
                return self.index == 1

            def release(self):
                pass

            def set(self, _prop, _value):
                return True

            def get(self, prop):
                return 640 if prop == 3 else 480

            def read(self):
                return True, np.full((48, 64, 3), 80, dtype=np.uint8)

        with patch("finger_lens.cv2.VideoCapture", side_effect=FakeCapture), patch(
            "finger_lens.camera_backend_candidates", return_value=[("test", 0)]
        ):
            capture = open_camera(-1, 640, 480)

        self.assertEqual(opened_indices, [0, 1])
        self.assertEqual(capture.index, 1)

    @patch("finger_lens.cv2.getWindowProperty", return_value=0.0)
    def test_native_close_button_is_detected(self, _get_window_property):
        self.assertTrue(window_is_closed("FingerLens"))

    @patch("finger_lens.cv2.getWindowProperty", return_value=1.0)
    def test_visible_window_remains_open(self, _get_window_property):
        self.assertFalse(window_is_closed("FingerLens"))

    def test_platform_camera_help(self):
        self.assertIn("Windows 设置", camera_help("Windows"))
        self.assertIn("系统设置", camera_help("Darwin"))
        self.assertIn("/dev/video", camera_help("Linux"))

    def test_clap_then_release_triggers_once(self):
        def hand(center_x):
            points = np.full((21, 2), (center_x, 70.0), dtype=np.float32)
            points[5] = (center_x - 20, 70)
            points[17] = (center_x + 20, 70)
            return points

        switcher = ClapCycleSwitcher(stable_frames=3, release_frames=3)
        close_hands = {"Left": hand(80), "Right": hand(120)}
        apart_hands = {"Left": hand(30), "Right": hand(190)}
        self.assertFalse(switcher.update(close_hands))
        self.assertFalse(switcher.update(close_hands))
        self.assertFalse(switcher.update(close_hands))
        self.assertTrue(switcher.armed)
        self.assertFalse(switcher.update(apart_hands))
        self.assertFalse(switcher.update(apart_hands))
        self.assertTrue(switcher.update(apart_hands))
        self.assertFalse(switcher.update(apart_hands))

    def test_clap_survives_temporary_hand_loss(self):
        def hand(center_x):
            points = np.full((21, 2), (center_x, 70.0), dtype=np.float32)
            points[0] = (center_x, 95)
            points[5] = (center_x - 20, 70)
            points[17] = (center_x + 20, 70)
            return points

        switcher = ClapCycleSwitcher(stable_frames=1, release_frames=2)
        close_hands = {"Left": hand(80), "Right": hand(120)}
        apart_hands = {"Left": hand(25), "Right": hand(195)}
        self.assertFalse(switcher.update(close_hands))
        self.assertTrue(switcher.armed)
        self.assertFalse(switcher.update({"Left": hand(100)}))
        self.assertFalse(switcher.update(apart_hands))
        self.assertTrue(switcher.update(apart_hands))


if __name__ == "__main__":
    unittest.main()
