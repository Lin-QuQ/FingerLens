import unittest

import numpy as np

from finger_lens import (
    ClapCycleSwitcher,
    SmoothLandmarks,
    FILTER_NAMES,
    camera_backend_candidates,
    camera_frame_is_black,
    camera_help,
    draw_zones,
    fashion_filter,
    polygon_mask,
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

    def test_two_hands_change_zone_pixels(self):
        frame = np.full((160, 220, 3), 80, dtype=np.uint8)
        left = np.tile(np.array([30.0, 80.0]), (21, 1))
        right = np.tile(np.array([190.0, 80.0]), (21, 1))
        for i, tip in enumerate((4, 8, 12, 16, 20)):
            left[tip] = (30 + i * 8, 30 + i * 24)
            right[tip] = (190 - i * 8, 30 + i * 24)
        result = draw_zones(frame, {"Left": left, "Right": right}, 1.0, 1)
        self.assertGreater(np.mean(np.abs(result.astype(int) - frame.astype(int))), 1.0)

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
