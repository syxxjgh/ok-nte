"""出售小红 (auction/sell_red.py) 纯逻辑单测: 聚类识别, 出售判定, 名称规范
化与屏内去重。合成帧/合成输入直接构造, 不依赖游戏画面; UI 流程
(sell_small_red_items) 属人工验证范围。"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from ok import Box

from src.tasks.AutoBidAuctionTask import AutoBidAuctionTask
from src.tasks.auction import bid as auction_bid
from src.tasks.auction import bid_price as auction_bid_price
from src.tasks.auction import layout as auction_layout
from src.tasks.auction import match as auction_match
from src.tasks.auction import options as auction_options
from src.tasks.auction import price as auction_price
from src.tasks.auction import sell_red

# 红色卡片主色 (BGR 序): 2026-10-28 实机截图标定 RGB(105,9,34) 的卡框色。
BGR_RED = (34, 9, 105)
BGR_BORDER = (60, 24, 170)  # 卡片外圈亮粉红描边, 在 RED_CARD_COLOR_RANGE 之外
BGR_PATTERN = (200, 220, 240)  # 已知红中间图案亮色
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


class TestItemIdentity(unittest.TestCase):
    GRID = Box(100, 200, 600, 300)

    def test_same_item_same_identity_across_scroll(self):
        # 同一件藏品: 滚动前在视口底部, 滚动后滑块下移、条目上移, 指纹不变。
        before = sell_red.item_identity((300, 350), self.GRID, (0, 100, 400))
        after = sell_red.item_identity((300, 200), self.GRID, (50, 100, 400))
        self.assertIsNotNone(before)
        self.assertEqual(before, after)

    def test_different_items_different_identity(self):
        first = sell_red.item_identity((300, 250), self.GRID, (0, 100, 400))
        second = sell_red.item_identity((300, 350), self.GRID, (0, 100, 400))
        self.assertNotEqual(first, second)

    def test_none_geometry_returns_none(self):
        self.assertIsNone(sell_red.item_identity((300, 250), self.GRID, None))

    def test_x_bucket_absorbs_jitter(self):
        first = sell_red.item_identity((302, 250), self.GRID, (0, 100, 400))
        second = sell_red.item_identity((318, 250), self.GRID, (0, 100, 400))
        self.assertEqual(first, second)


class TestIdentityKnown(unittest.TestCase):
    def test_exact_and_tolerance_match(self):
        known = {(10, 0.50)}
        self.assertTrue(sell_red.identity_known((10, 0.50), known))
        self.assertTrue(sell_red.identity_known((10, 0.53), known))  # 容差内

    def test_different_bucket_not_matched(self):
        known = {(10, 0.50)}
        self.assertFalse(sell_red.identity_known((11, 0.50), known))

    def test_none_identity_never_known(self):
        self.assertFalse(sell_red.identity_known(None, {(10, 0.5)}))


class TestNameKnown(unittest.TestCase):
    def test_substring_both_directions(self):
        self.assertTrue(sell_red._name_known("霜铁交响", {"霜铁交响"}))
        self.assertTrue(sell_red._name_known("霜铁", {"霜铁交响"}))  # OCR 丢字
        self.assertTrue(sell_red._name_known("霜铁交响曲", {"霜铁交响"}))  # OCR 多字

    def test_unknown_name(self):
        self.assertFalse(sell_red._name_known("白龙王", {"霜铁交响"}))
        self.assertFalse(sell_red._name_known("", {"霜铁交响"}))


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


class TestUnknownRedClusters(unittest.TestCase):
    GRID = Box(20, 20, 560, 560)

    def test_detects_unknown_1x1_and_1x2_rejects_known_and_thin_sliver(self):
        frame = make_frame(620, 620)
        # 1x1 纯红未知红
        paint(frame, 40, 40, 55, 55, BGR_RED)
        # 1x2 纯红未知红(轮廓已知但无图案)
        paint(frame, 40, 120, 55, 120, BGR_RED)
        # 已知红(外圈红框 + 内部大半亮色图案)
        paint(frame, 140, 40, 120, 120, BGR_RED)
        paint(frame, 165, 65, 70, 70, BGR_PATTERN)
        # 边缘细条(宽不足 min_side)
        paint(frame, 560, 300, 10, 80, BGR_RED)

        clusters = sell_red.unknown_red_clusters(frame, self.GRID)
        self.assertEqual(len(clusters), 2)
        self.assertEqual((clusters[0][2], clusters[0][3]), (55, 55))
        self.assertEqual((clusters[1][2], clusters[1][3]), (55, 120))

    def test_adjacent_unknown_reds_with_bright_border_stay_separate(self):
        frame = make_frame(620, 620)
        # 相邻两张 1x1 未知红, 中间用 4px 亮粉红描边隔开
        paint(frame, 40, 40, 58, 58, BGR_BORDER)
        paint(frame, 44, 44, 50, 50, BGR_RED)
        paint(frame, 98, 40, 58, 58, BGR_BORDER)
        paint(frame, 102, 44, 50, 50, BGR_RED)

        clusters = sell_red.unknown_red_clusters(frame, self.GRID)
        self.assertEqual(len(clusters), 2)


class TestShowcaseScrollAndDedup(unittest.TestCase):
    def test_estimate_vertical_shift_on_scrolled_texture(self):
        # 构造高 600px 的虚拟展柜内容, 用多块不同颜色卡片提供纹理
        canvas = make_frame(400, 600)
        paint(canvas, 30, 30, 60, 60, BGR_RED)
        paint(canvas, 150, 80, 90, 90, (180, 120, 40))
        paint(canvas, 60, 210, 70, 110, (50, 160, 80))
        paint(canvas, 220, 250, 110, 80, (40, 130, 190))
        paint(canvas, 100, 380, 80, 60, BGR_RED)

        prev_crop = canvas[0:360, :]
        curr_crop = canvas[140:500, :]
        self.assertEqual(sell_red.estimate_vertical_shift(prev_crop, prev_crop), 0)
        self.assertEqual(sell_red.estimate_vertical_shift(prev_crop, curr_crop), 140)

    def test_merge_unknown_red_cluster_deduplicates_across_scroll(self):
        known: list[tuple[int, int, int, int]] = []
        self.assertTrue(sell_red.merge_unknown_red_cluster(known, (100, 220, 55, 55), cum_dy=0))
        # 下滚 120px 后, 同一件藏品在视口中的 y 变成 100, 绝对坐标仍为 220
        self.assertFalse(sell_red.merge_unknown_red_cluster(known, (102, 100, 55, 55), cum_dy=120))
        # 另一件不同位置的藏品应被计入
        self.assertTrue(sell_red.merge_unknown_red_cluster(known, (100, 280, 55, 55), cum_dy=120))
        self.assertEqual(len(known), 2)

    def test_scan_showcase_unknown_reds_counts_across_two_screens(self):
        canvas = make_frame(420, 600)
        # 第 1 屏可见: 2 个未知红(第 2 个在 y=240 位于两屏重叠区), 第 2 屏新增 1 个(y=420)
        paint(canvas, 30, 40, 55, 55, BGR_RED)
        paint(canvas, 140, 220, 55, 55, BGR_RED)
        paint(canvas, 220, 110, 90, 80, (180, 120, 40))  # 纹理参照物
        paint(canvas, 40, 430, 55, 55, BGR_RED)
        paint(canvas, 210, 320, 90, 90, (50, 160, 80))

        def make_screen(offset_y: int, thumb_top: int) -> np.ndarray:
            full = make_frame(500, 400)
            full[20:340, 20:440] = canvas[offset_y : offset_y + 320, :]
            # 滚动条放在 x=450..462, y=20..340 (轨道高 320)
            full[20:29, 450:462] = (200, 200, 200)
            full[331:340, 450:462] = (200, 200, 200)
            full[20 + thumb_top : 20 + thumb_top + 140, 450:462] = (185, 185, 185)
            return full

        frames = [
            make_screen(0, 9),      # 初始检测在顶并扫第 1 屏
            make_screen(180, 171),  # 下滚一次后扫第 2 屏并判定滑块贴底结束
        ]

        class FakeTask:
            def __init__(self):
                self.idx = -1
                self.frame = None
                self.infos = []
                self.debugs = []
                self.scrolls = []
                self.events = []

            def next_frame(self):
                self.events.append("next_frame")
                self.idx = min(self.idx + 1, len(frames) - 1)
                self.frame = frames[self.idx]
                return self.frame

            def box_of_screen(self, x, y, to_x=1.0, to_y=1.0):
                return Box(round(x * 500), round(y * 400), round((to_x - x) * 500), round((to_y - y) * 400))

            def scroll(self, x, y, count):
                self.events.append(("scroll", count))
                self.scrolls.append(count)

            def _bounded_sleep(self, deadline, delay):
                self.events.append(("sleep", delay))

            def log_info(self, msg):
                self.infos.append(msg)

            def log_debug(self, msg):
                self.debugs.append(msg)

        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        boxes = replace(
            boxes,
            showcase_grid=Box(20, 20, 420, 320),
            showcase_scrollbar=Box(450, 20, 12, 320),
        )
        task = FakeTask()
        count = sell_red.scan_showcase_unknown_reds(task, boxes)
        self.assertEqual(count, 3)
        self.assertEqual(task.events[0], ("sleep", sell_red.SHOWCASE_SCAN_PRE_DELAY))
        self.assertEqual(task.events[1], "next_frame")
        # 先向下滚动扫描, 结束前再用正滚轮把 Y 轴滚动条复原到顶部
        self.assertEqual(task.scrolls[0], sell_red.SHOWCASE_SCROLL_WHEEL)
        self.assertGreater(task.scrolls[-1], 0)
        self.assertEqual(len(task.scrolls), 1 + sell_red.SHOWCASE_SCROLL_TOP_PASSES)

    def test_scan_showcase_unknown_reds_continues_when_shift_zero_on_pass1(self):
        top_frame = make_frame(500, 400)
        # 顶屏只有 1 个未知红, 无额外参照纹理
        paint(top_frame, 50, 50, 55, 55, BGR_RED)
        # 滚动条在顶部
        top_frame[20:29, 450:462] = (200, 200, 200)
        top_frame[331:340, 450:462] = (200, 200, 200)
        top_frame[29:169, 450:462] = (185, 185, 185)

        bottom_frame = make_frame(500, 400)
        # 底屏新增 2 个不同位置的未知红(含无纹理/大步跳跃的场景)
        paint(bottom_frame, 50, 220, 55, 55, BGR_RED)
        paint(bottom_frame, 180, 220, 55, 55, BGR_RED)
        # 滚动条贴底
        bottom_frame[20:29, 450:462] = (200, 200, 200)
        bottom_frame[331:340, 450:462] = (200, 200, 200)
        bottom_frame[191:331, 450:462] = (185, 185, 185)

        frames = [top_frame, bottom_frame]

        class FakeTask:
            def __init__(self):
                self.idx = -1
                self.frame = None
                self.scrolls = []

            def next_frame(self):
                self.idx = min(self.idx + 1, len(frames) - 1)
                self.frame = frames[self.idx]
                return self.frame

            def box_of_screen(self, x, y, to_x=1.0, to_y=1.0):
                return Box(round(x * 500), round(y * 400), round((to_x - x) * 500), round((to_y - y) * 400))

            def scroll(self, x, y, count):
                self.scrolls.append(count)

            def _bounded_sleep(self, deadline, delay):
                pass

            def log_info(self, msg):
                pass

            def log_debug(self, msg):
                pass

        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        boxes = replace(
            boxes,
            showcase_grid=Box(20, 20, 420, 320),
            showcase_scrollbar=Box(450, 20, 12, 320),
        )
        task = FakeTask()
        with patch.object(sell_red, "estimate_vertical_shift", return_value=0):
            count = sell_red.scan_showcase_unknown_reds(task, boxes)
        self.assertEqual(count, 3)
        self.assertEqual(task.scrolls[0], sell_red.SHOWCASE_SCROLL_WHEEL)
        self.assertGreater(task.scrolls[-1], 0)


class TestUnknownRedSnipePricing(unittest.TestCase):
    def test_parse_left_highest_bid_handles_normal_round1_and_dark_mode(self):
        # 天黑了模式: 仅名次(1~4)和槽位号(3/5), 忽略并返 None
        self.assertIsNone(
            auction_price.parse_left_highest_bid(["lllmarco", "好想玩原神", "2", "3", "5", "小姜"])
        )
        # 第 1 回合未结算: 全部为 0 或空
        self.assertIsNone(auction_price.parse_left_highest_bid(["小姜", "0", "无垠"]))
        # 第 2/4 回合正常金额: 取单个最大金额并忽略用户名里的 K/M 字母和槽位数字
        self.assertEqual(
            auction_price.parse_left_highest_bid(
                ["小姜", "Player99M", "888.89K", "155.56K", "1.02M", "666.67K", "1.44M", "990K", "5"]
            ),
            1440000,
        )

    def test_snipe_price_with_floor_repeats_increment_10000(self):
        self.assertEqual(auction_price.snipe_price_with_floor(1500000, None), 1500000)
        self.assertEqual(auction_price.snipe_price_with_floor(1500000, 800000), 1500000)
        self.assertEqual(auction_price.snipe_price_with_floor(1500000, 1500000), 1510000)
        self.assertEqual(auction_price.snipe_price_with_floor(1500000, 1510000), 1520000)
        self.assertEqual(auction_price.snipe_price_with_floor(1500000, 1500000, 50000), 1550000)
        self.assertEqual(auction_price.snipe_price_with_floor(1500000, 1550000, 50000), 1600000)
        # 左侧最高出价 + 加价高于固定秒仓价时, 在左侧最高基础上加价
        self.assertEqual(
            auction_price.snipe_price_with_floor(1500000, 1500000, 50000, left_highest=1800000),
            1850000,
        )
        # 左侧最高出价 + 加价低于固定秒仓价时, 仍出固定秒仓价
        self.assertEqual(
            auction_price.snipe_price_with_floor(1500000, None, 50000, left_highest=888890),
            1500000,
        )

    def test_validate_estimate_checks_snipe_price_only_when_enabled(self):
        task = SimpleNamespace(
            config={
                auction_options.CONF_TIERED_RATIO: False,
                auction_options.CONF_FIXED_PRICE: 500000,
                auction_options.CONF_SNIPE_UNKNOWN_RED: True,
                auction_options.CONF_SNIPE_PRICE: 0,
                auction_options.CONF_SNIPE_INCREMENT: 10000,
            },
            log_warning=lambda msg: None,
        )
        with self.assertRaises(ValueError):
            auction_bid_price._validate_estimate(task)
        task.config[auction_options.CONF_SNIPE_PRICE] = 1500000
        task.config[auction_options.CONF_SNIPE_INCREMENT] = 0
        with self.assertRaises(ValueError):
            auction_bid_price._validate_estimate(task)
        task.config[auction_options.CONF_SNIPE_UNKNOWN_RED] = False
        auction_bid_price._validate_estimate(task)
        task.config[auction_options.CONF_SNIPE_UNKNOWN_RED] = True
        task.config[auction_options.CONF_SNIPE_INCREMENT] = 50000
        auction_bid_price._validate_estimate(task)

    def test_price_estimate_ignores_estimate_base_and_heart_when_snipe_triggered(self):
        task = SimpleNamespace(
            config={
                auction_options.CONF_BID_MODE: auction_options.BID_MODE_ESTIMATE,
                auction_options.CONF_FIXED_PRICE: 500000,
                auction_options.CONF_HEART_INCREMENT: 666666,
                auction_options.CONF_SNIPE_UNKNOWN_RED: True,
                auction_options.CONF_SNIPE_PRICE: 1500000,
                auction_options.CONF_SNIPE_INCREMENT: 50000,
            },
            _snipe_triggered=True,
            _snipe_left_highest=None,
            _heart_present=True,
            _asset_routed_mode=None,
            last_bid_price=None,
            current_bid_count=0,
            _bid_mode=lambda: auction_options.BID_MODE_ESTIMATE,
            log_info=lambda msg: None,
            log_warning=lambda msg: None,
        )
        # 触发秒仓时不读估价、不加基础价、不叠永恒之心加价
        self.assertEqual(auction_bid_price.calculate_price(task), 1500000)
        # 下一回合未拿下继续出价时: 天黑了模式(左侧为 None)按自身上轮出价加价
        task.last_bid_price = 1500000
        task.current_bid_count = 1
        self.assertEqual(auction_bid_price.calculate_price(task), 1550000)
        # 正常模式读到左侧最高 1.80M 时, 直接在 1.80M 基础上加 50000
        task.last_bid_price = 1550000
        task._snipe_left_highest = 1800000
        task.current_bid_count = 2
        self.assertEqual(auction_bid_price.calculate_price(task), 1850000)

    def test_task_check_unknown_red_snipe_rules_and_retry_dedup(self):
        task = object.__new__(AutoBidAuctionTask)
        task.config = {
            auction_options.CONF_BID_MODE: auction_options.BID_MODE_ESTIMATE,
            auction_options.CONF_SNIPE_UNKNOWN_RED: True,
        }
        task._asset_routed_mode = None
        task.current_bid_count = 0
        task._snipe_triggered = False
        task._snipe_checked_bid = 0
        task._snipe_left_highest = None
        task._aux_error_count = 0
        task.next_frame = lambda: None
        ocr_tokens = iter([["888.89K", "660K"], ["1.44M", "990K"]])
        task.ocr = lambda **kw: [Box(0, 0, 10, 10, name=t) for t in next(ocr_tokens, [])]
        task.log_info = lambda msg: None
        task.log_warning = lambda msg: None
        task.log_debug = lambda msg: None
        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))

        # 第 1 回合数量不足 3 时保持不触发
        with patch.object(sell_red, "scan_showcase_unknown_reds", return_value=2) as mock_scan:
            task._check_unknown_red_snipe(boxes)
            self.assertFalse(task._snipe_triggered)
            self.assertIsNone(task._snipe_left_highest)
            self.assertEqual(mock_scan.call_count, 1)

        # 第 2 回合命中 3 个未知红后触发秒仓并读取左侧最高出价, 同口重试不重复扫描
        task.current_bid_count = 1
        with patch.object(sell_red, "scan_showcase_unknown_reds", return_value=3) as mock_scan:
            task._check_unknown_red_snipe(boxes)
            self.assertTrue(task._snipe_triggered)
            self.assertEqual(task._snipe_left_highest, 888890)
            task._check_unknown_red_snipe(boxes)
            self.assertTrue(task._snipe_triggered)
            self.assertEqual(mock_scan.call_count, 1)

        # 第 3 回合延续秒仓锁存状态, 更新左侧最高出价而不重扫展柜
        task.current_bid_count = 2
        with patch.object(sell_red, "scan_showcase_unknown_reds", return_value=0) as mock_scan:
            task._check_unknown_red_snipe(boxes)
            self.assertTrue(task._snipe_triggered)
            self.assertEqual(task._snipe_left_highest, 1440000)
            mock_scan.assert_not_called()

        # 第 4 回合(>3)不扫描且退出秒仓状态
        task.current_bid_count = 3
        with patch.object(sell_red, "scan_showcase_unknown_reds", return_value=5) as mock_scan:
            task._check_unknown_red_snipe(boxes)
            self.assertFalse(task._snipe_triggered)
            self.assertIsNone(task._snipe_left_highest)
            mock_scan.assert_not_called()

    def test_attempt_bid_scans_showcase_after_instrument_and_before_keypad(self):
        events: list[str] = []
        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        task = SimpleNamespace(
            _reading_ops=lambda: SimpleNamespace(asset_once=lambda *_: 5000000),
            _remaining_timeout=lambda deadline, limit, message="": limit,
            _check_heart_once=lambda b: events.append("heart"),
            _should_abandon_without_heart=lambda: False,
            _use_instrument_once=lambda b, d: events.append("instrument"),
            _check_unknown_red_snipe=lambda b: events.append("snipe"),
            _bid_panel_open=lambda b: False,
            _wait_operate_click=lambda *a, **kw: True,
            wait_ocr=lambda *a, **kw: [Box(0, 0, 10, 10)],
            _calculate_auction_price=lambda b, d: (events.append("price"), 1500000)[1],
            _input_fixed_price=lambda b, price, deadline: events.append(f"input:{price}"),
            wait_until=lambda *a, **kw: True,
            _send_emote_once=lambda: events.append("emote"),
            log_info=lambda msg: None,
            log_warning=lambda msg: None,
            log_debug=lambda msg: None,
        )
        self.assertTrue(auction_bid.attempt_bid(task, boxes, deadline=999999.0))
        self.assertEqual(events, ["heart", "instrument", "snipe", "price", "input:1500000", "emote"])

    def _make_venue_task(self, expected: str):
        task = object.__new__(AutoBidAuctionTask)
        task.config = {auction_options.CONF_EXPECTED_VENUE: expected}
        task.info_set = MagicMock()
        task.stop_rounds = MagicMock()
        task.log_error = MagicMock()
        task._read_current_venue = MagicMock(return_value="海贝场")
        return task

    def test_check_expected_venue_mismatch_stops_and_raises(self):
        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        task = self._make_venue_task("真珠场")

        with self.assertRaises(ValueError):
            task._check_expected_venue(boxes, venue="海贝场")

        task.info_set.assert_called_once_with("当前会场", "海贝场")
        task.stop_rounds.assert_called_once()
        task.log_error.assert_called_once()
        self.assertTrue(task.log_error.call_args.kwargs.get("notify"))

    def test_check_expected_venue_match_passes(self):
        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        task = self._make_venue_task("真珠")

        task._check_expected_venue(boxes, venue="真珠场")

        task.info_set.assert_called_once_with("当前会场", "真珠场")
        task.stop_rounds.assert_not_called()
        task.log_error.assert_not_called()

    def test_check_expected_venue_empty_expected_or_venue_passes(self):
        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        task_no_expected = self._make_venue_task("")
        task_no_expected._check_expected_venue(boxes, venue="海贝场")
        task_no_expected.stop_rounds.assert_not_called()
        task_no_expected.log_error.assert_not_called()

        task_no_venue = self._make_venue_task("真珠场")
        task_no_venue._check_expected_venue(boxes, venue="")
        task_no_venue.info_set.assert_called_once_with("当前会场", "未识别")
        task_no_venue.stop_rounds.assert_not_called()
        task_no_venue.log_error.assert_not_called()

    def test_handle_match_click_checks_venue_before_clicking_match(self):
        boxes = auction_layout.build_boxes(lambda x, y, to_x=1.0, to_y=1.0: Box(0, 0, 10, 10))
        events: list[str] = []

        def fail_venue(_boxes):
            events.append("check_venue")
            raise ValueError("会场不匹配")

        actions = auction_match.MatchActions(
            is_bid_screen=lambda b: False,
            is_confirm_screen=lambda b: False,
            is_skip_screen=lambda b: False,
            is_match_screen=lambda b: True,
            is_world_screen=lambda: False,
            dismiss_notice=lambda *a, **kw: False,
            next_frame=lambda: None,
            sleep=lambda s: None,
            wait_operate_click=lambda *a, **kw: (events.append("click_match"), True)[1],
            wait_ocr=lambda *a, **kw: [],
            wait_until=lambda *a, **kw: False,
            operate_click=lambda *a, **kw: None,
            remaining_timeout=lambda d, l, m="": l,
            resume_after_world_drop=lambda b, d: auction_layout.AuctionState.SKIP,
            poll_interval=0.1,
            log_info=lambda msg: None,
            log_debug=lambda msg: None,
            log_warning=lambda msg: None,
            check_expected_venue=fail_venue,
        )

        with self.assertRaises(ValueError):
            auction_match.handle_match_click(actions, boxes, stage_deadline=999999.0)
        self.assertEqual(events, ["check_venue"])


if __name__ == "__main__":
    unittest.main()
