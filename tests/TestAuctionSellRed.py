"""出售小红 (auction/sell_red.py) 纯逻辑单测: 聚类识别, 出售判定, 名称规范
化与屏内去重。合成帧/合成输入直接构造, 不依赖游戏画面; UI 流程
(sell_small_red_items) 属人工验证范围。"""

import unittest

import numpy as np
from ok import Box

from src.tasks.auction import sell_red

# 红色卡片主色 (BGR 序): 2026-10-28 实机截图标定 RGB(105,9,34) 的卡框色。
BGR_RED = (34, 9, 105)
BGR_FAR = (30, 30, 30)


def make_frame(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), BGR_FAR, dtype=np.uint8)


def paint(frame: np.ndarray, x: int, y: int, w: int, h: int, color) -> None:
    frame[y : y + h, x : x + w] = color


class TestRedItemClusters(unittest.TestCase):
    GRID = Box(10, 20, 180, 120)

    def test_two_cards_sorted_top_to_bottom(self):
        frame = make_frame(200, 150)
        paint(frame, 20, 30, 40, 30, BGR_RED)  # 上排
        paint(frame, 120, 90, 40, 30, BGR_RED)  # 下排
        clusters = sell_red.red_item_clusters(frame, self.GRID)
        self.assertEqual(len(clusters), 2)
        # 矩形绝对中心 (40,45)/(140,105), 质心 int 截断半像素 → 39/44 与 139/104。
        self.assertEqual(clusters[0], (39, 44))
        self.assertEqual(clusters[1], (139, 104))
        self.assertLess(clusters[0][1], clusters[1][1])  # 从上到下

    def test_noise_below_area_threshold_is_dropped(self):
        frame = make_frame(200, 150)
        paint(frame, 20, 30, 40, 30, BGR_RED)
        paint(frame, 100, 40, 3, 3, BGR_RED)  # 远小于面积阈值
        clusters = sell_red.red_item_clusters(frame, self.GRID)
        self.assertEqual(len(clusters), 1)

    def test_out_of_range_color_not_matched(self):
        frame = make_frame(200, 150)  # 全是底色, 无红卡
        self.assertEqual(sell_red.red_item_clusters(frame, self.GRID), [])

    def test_empty_crop_returns_empty(self):
        frame = make_frame(200, 150)
        outside = Box(500, 500, 100, 100)  # 帧外
        self.assertEqual(sell_red.red_item_clusters(frame, outside), [])


class TestScrollbarThumbRatio(unittest.TestCase):
    STRIP = Box(0, 0, 12, 400)

    def make_bar(self, thumb_top: int, thumb_bottom: int):
        """合成滚动条: 暗轨道 + 两端亮端帽(9px) + 浅灰滑块。"""
        frame = np.full((400, 12, 3), (25, 25, 25), dtype=np.uint8)
        frame[0:9] = (200, 200, 200)
        frame[391:400] = (200, 200, 200)
        frame[thumb_top:thumb_bottom] = (185, 185, 185)
        return frame

    def test_bottom_position_is_one(self):
        frame = self.make_bar(330, 391)  # 滑块贴底(含端帽行)
        ratio = sell_red.scrollbar_thumb_ratio(frame, self.STRIP)
        self.assertIsNotNone(ratio)
        self.assertGreaterEqual(ratio, 0.93)

    def test_top_position_is_near_zero(self):
        frame = self.make_bar(9, 140)
        ratio = sell_red.scrollbar_thumb_ratio(frame, self.STRIP)
        self.assertIsNotNone(ratio)
        self.assertLess(ratio, 0.4)

    def test_mid_position(self):
        frame = self.make_bar(200, 330)
        ratio = sell_red.scrollbar_thumb_ratio(frame, self.STRIP)
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 330 / 399, delta=0.03)

    def test_no_scrollbar_returns_none(self):
        frame = np.full((400, 12, 3), (25, 25, 25), dtype=np.uint8)  # 无亮行
        self.assertIsNone(sell_red.scrollbar_thumb_ratio(frame, self.STRIP))

    def test_empty_crop_returns_none(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        outside = Box(50, 50, 12, 400)  # 帧外
        self.assertIsNone(sell_red.scrollbar_thumb_ratio(frame, outside))


class TestShouldSellItem(unittest.TestCase):
    def test_none_price_keeps(self):
        self.assertFalse(sell_red.should_sell_item(None, 500000))

    def test_zero_or_negative_keeps(self):
        self.assertFalse(sell_red.should_sell_item(0, 500000))
        self.assertFalse(sell_red.should_sell_item(-1, 500000))

    def test_equal_limit_keeps(self):
        self.assertFalse(sell_red.should_sell_item(500000, 500000))

    def test_below_limit_sells(self):
        self.assertTrue(sell_red.should_sell_item(499999, 500000))
        self.assertTrue(sell_red.should_sell_item(1, 500000))


class TestNormalizeItemName(unittest.TestCase):
    def test_joins_and_strips_whitespace(self):
        self.assertEqual(sell_red.normalize_item_name(["白 龙 王", ""]), "白龙王")

    def test_empty_is_empty_string(self):
        self.assertEqual(sell_red.normalize_item_name([]), "")


class TestPickUnvisited(unittest.TestCase):
    def test_returns_first_outside_radius(self):
        clusters = [(100, 100), (300, 100)]
        self.assertEqual(
            sell_red.pick_unvisited(clusters, [(100, 105)], 10), (300, 100)
        )

    def test_all_skipped_returns_none(self):
        clusters = [(100, 100), (300, 100)]
        self.assertIsNone(
            sell_red.pick_unvisited(clusters, [(100, 100), (299, 101)], 10)
        )

    def test_empty_clusters_returns_none(self):
        self.assertIsNone(sell_red.pick_unvisited([], [], 10))


if __name__ == "__main__":
    unittest.main()
