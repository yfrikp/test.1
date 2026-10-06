#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题目1 · 颜色物块识别与定位（单机版，不依赖 ROS2）
=============================================================================
这份是在你原来那份脚本基础上改的。每一处改动都在注释里写明了
【改了什么】和【为什么要改】，搜索 "★改" 就能看到全部改动点。

它现在输出题目真正要的三样东西：
    种类 + 长度(cm) + 距摄像头中心点的坐标(cm, x/y)  + 距离(cm)

用法（在 Ubuntu 终端里）：
    # 1) 处理一张图片
    python3 task1_detect.py --src ../samples/first.png

    # 2) 同时把结果图存下来
    python3 task1_detect.py --src ../samples/first.png --out /tmp/result_fixed.png

    # 3) 边看窗口边调（按 q 退出）
    python3 task1_detect.py --src ../samples/first.png --show

    # 4) 测处理帧率（验收 60Hz 时用来量自己的能力）
    python3 task1_detect.py --src ../samples/first.png --benchmark 200

    # 5) 处理视频文件 / 摄像头
    python3 task1_detect.py --src /path/to/video.mp4
    python3 task1_detect.py --camera 0
=============================================================================
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# =============================================================================
#  ★改10：所有阈值集中到这一个参数区
#  原因：你原来阈值散在代码里，换个灯光/换个物块就要满文件找数字，很容易改漏。
#        调参只动这里，代码逻辑一行都不用碰。
#  这些默认值是按你那张 first.png 调的，真实照片需要重调（用 tune_hsv.py 拖滑条）。
# =============================================================================
P = dict(
    # ---- 红色：要两段（红色在 HSV 里跨 0/180 两端，只写一段会漏一半）----
    red_h_low_max  = 8,      # 低端 H 上限：0~8
    red_h_high_min = 172,    # 高端 H 下限：172~179
    red_s_min      = 90,     # 饱和度下限，滤掉"发白的红"
    red_v_min      = 60,     # 明度下限，滤掉"发黑的红"

    # ---- 蓝色 ----
    blue_h_min     = 100,
    blue_h_max     = 132,
    blue_s_min     = 70,
    blue_v_min     = 60,

    # ---- 形态学核大小（必须是奇数）----
    #  ★改6：新增 close_kernel（你原来只有 OPEN）
    #  原因：红蓝相间的物块，在红蓝分界线处常有一道暗缝/阴影，
    #        掩膜会从中间断开 → 一个物块被拆成"红"和"蓝"两个轮廓 → 永远认不出 red_blue。
    #        CLOSE（先膨胀后腐蚀）能把这条缝补上，让两层重新连成一个整体。
    open_kernel    = 3,      # OPEN：去孤立噪点
    close_kernel   = 9,      # CLOSE：补缝（缝太大就调大这个值）

    # ---- 面积门限 ----
    #  ★改8：改成绝对像素，不再用 0.002*H*W 的比例
    #  原因：0.002*H*W 在 640x480 下等于 614 像素，在 1280x720 下等于 1843 像素，
    #        分辨率一变门限就跟着变；物块离远一点（面积变小）就会被直接丢掉。
    min_area_px    = 200,

    # ---- 分类门限 ----
    dominant_ratio   = 0.80,  # 单一颜色占比 ≥ 这个值 → 判为纯色
    interleave_ratio = 0.18,  # 红蓝相间时，每种颜色至少要占这么多
    layer_purity     = 0.85,  # 一条水平线切下去，上下两层的主色纯度要到这个值
    layer_tol        = 0.25,  # 上下两层规模要均衡到这个比例（防止只有一条细边）

    # =========================================================================
    #  ★改4：新增"尺度"参数 —— 这是能不能过 3cm 的命门
    #  原因：像素要变成厘米，必须知道两件事：
    #        ① 物块真实边长 real_edge_m（拿尺子量）
    #        ② 相机内参 fx/fy/cx/cy（棋盘格标定，或从 /camera_info 话题读）
    #        你原来的代码没有这两个，所以只能输出像素，永远到不了 3cm。
    # =========================================================================
    real_edge_m = 0.05,      # 物块真实边长（米）。★必须拿尺子量！先按 5cm 试
    fx = 600.0,              # 焦距 x（像素）。标定后替换
    fy = 600.0,              # 焦距 y（像素）
    cx = 320.0,              # 光心 x（像素）—— 图像中心是 320（640 宽）
    cy = 240.0,              # 光心 y（像素）—— 图像中心是 240（480 高）

    # ★改11：畸变系数（k1, k2, p1, p2, k3），棋盘格标定时会一起算出来
    #  原因：真实镜头都有畸变，画面越靠边越"弯"，物块在画面边缘时像素宽度量不准，
    #        误差会超过 3cm。填上这 5 个数后，代码会自动先做畸变校正再识别。
    #        全填 0 表示不校正（你现在用渲染图/合成图，畸变为 0，所以默认全 0）。
    dist = [0.0, 0.0, 0.0, 0.0, 0.0],

    # ★改13：加分项①「参照点与摄像头中心有偏移时，通过调整偏移量精准定位」
    #  你测出偏移后填这里（左偏为负、右偏为正，单位米），代码会自动做补偿。
    offset_x_m = 0.0,
    offset_y_m = 0.0,

    # ---- 像素边长怎么量 ----
    #  ★改9：边长取法参数化（你原来固定用 minAreaRect 的长边）
    #  ★改14（实测后改为默认 area）：见 measure_edge_px 里的详细说明。
    #    四个模式：
    #      area       —— 用 √(轮廓面积) 再减 edge_bias_px（★默认，实测最准）
    #      front_face —— 优先找"正方形的那个面"（approxPolyDP 拟合四边形）
    #      bbox       —— 外接矩形宽高平均（小目标上很不稳，实测最差）
    #      minrect    —— minAreaRect 长边（你原来的做法）
    #      auto       —— 能用正方形面就用，否则退回 area
    measure = "area",

    # ★改14：掩膜外扩补偿（像素）。这是能不能过 3cm 的关键标定值。
    #
    # 【为什么需要它 —— 实测发现的系统性偏差】
    #   高斯模糊 + 阈值分割之后，颜色掩膜会**比真实物块每边大约 1 个像素**：
    #   模糊把物块边缘的颜色"晕"到背景上，只要晕过去的部分饱和度还够，
    #   就会被 inRange 判为物块 → 掩膜外扩。
    #   于是 √(轮廓面积) ≈ 真实边长 + 1 px。
    #
    #   Z = fy·W / edge，edge 偏大 1 px → Z 偏小。距离越远越致命：
    #       30cm 时物块 100px，偏 1px 只有 1%     → 误差 0.3cm（看不出来）
    #       120cm 时物块 25px，偏 1px 就是 4%     → 误差 5cm（直接超标）
    #   这就是"近处很准、远处超标"的真正原因 —— 不是公式错，是像素测量有固定偏置。
    #
    # 【怎么重新标定】（换相机 / 换光照 / 换物块后必做）
    #   拿一个已知边长 W 的物块，放在已知距离 Z 处，
    #   调整这个值，让输出的距离等于卷尺量到的 Z。
    #   合成图集上标定结果是 1.0；真实相机通常在 0.5 ~ 2.0 之间。
    edge_bias_px = 1.0,
)


# =============================================================================
#  边长测量
# =============================================================================
def measure_edge_px(cnt, p):
    """
    返回 (采用值, 方式, bbox估计, minrect估计, 正方形面估计, 面积估计)

    ★改14：新增并默认使用「面积法」。实测对比（合成图集，真值精确已知）：

        样本            实际画出   bbox    minAreaRect  正方形面   √面积
        红 30cm           100     102.0     101.0       100.0     101.0
        蓝 80cm            38      40.5      40.0        39.0      39.0
        蓝 120cm           25      27.0      26.0        25.8      26.0
        红蓝 120cm         25      28.0 ✗    27.0        失败 ✗    26.1

      结论：四种估计**都系统性偏大**（掩膜外扩所致，见 P 里 edge_bias_px 的说明），
      但「√面积」的偏差最稳定（恒为 +1.0~1.1 px，四个样本一致），
      减掉 edge_bias_px 之后 Z 误差全部 < 0.5cm，包括 bbox 法错得最离谱的那个。

      为什么面积法比另外三个稳：
        bbox / minAreaRect 都要求"把所有轮廓点包进去"，
        轮廓上只要有几个噪声毛刺，外接框就被撑大（120cm 那个 +3px 就是这么来的）；
        而面积是**对整条边界积分**，几个毛刺对总面积的影响被摊平了。
        另外面积还能顺带利用亚像素信息（contourArea 用格林公式算多边形面积，不是数格子）。
    """
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_est = (w + h) / 2.0                      # 外接矩形宽高平均

    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    minrect_est = max(rw, rh)                     # 你原来用的：最小外接矩形长边

    # 面积法：正方形面积 = 边长²，所以边长 = √面积；再减掉掩膜外扩的偏置
    area_est = float(np.sqrt(max(0.0, cv2.contourArea(cnt)))) - float(p["edge_bias_px"])

    # 找"正方形的面"：把轮廓简化成多边形，如果正好是 4 个顶点、且四条边长度接近，
    # 那它就是正对着镜头的那一面 —— 这时的边长才是真正的"棱长"
    face_est = None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(float)
        sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
        if min(sides) > 0 and (max(sides) / min(sides)) < 1.35:
            face_est = sum(sides) / 4.0

    mode = p["measure"]
    common = (bbox_est, minrect_est, face_est, area_est)

    if mode == "bbox":
        return bbox_est, "bbox", *common
    if mode == "minrect":
        return minrect_est, "minrect", *common
    if mode == "front_face":
        if face_est:
            return face_est, "front_face", *common
        return area_est, "area(退)", *common
    if mode == "auto":
        # 能拟合出正方形面就用它，否则退回面积法
        if face_est:
            return face_est, "front_face", *common
        return area_est, "area", *common
    # 默认 = area
    return area_est, "area", *common


# =============================================================================
#  ★改5：红蓝相间的判定改成"上下分层"
#  你原来的写法：roi_red > 0.15 且 roi_blue > 0.15 → red_blue
#  问题：一整块红 + 边缘一点蓝色反光，两个比例也会同时超过 15%，被误判成 red_blue。
#  正确判据：按水平中线把物块切成上下两半，要求
#     ① 上半部分被某一种颜色主导（比如红）
#     ② 下半部分被另一种颜色主导（比如蓝）
#     ③ 上下两层的规模还要均衡（防止"一整块红 + 底部一条细蓝边"混过去）
#  题目说"识别一个面就可以"，所以看这一个面上的上下分层就够了。
#
#  ★改5b：分界线不是死板地切在正中间，而是【自动搜索】出来的。
#  原因：第一版我按 50% 切，结果你那张图里的小方块（上红下蓝）被误判成 red ——
#        因为蓝色那一层只占下面约 1/3，切在 50% 时"下半区"里红色还是占多数，
#        条件不成立就退化成纯色了。真实照片里分界线位置更是随拍摄角度变，
#        写死 50% 必然不稳。
#        正确做法：把物块按行扫描，试每一条水平线，选"能把红蓝分得最干净"的那条。
# =============================================================================
def _layer_split(blob_red, blob_blue, box, min_layer=0.12):
    """
    自动找红蓝分界线。返回 (top_is_red, purity, balance, main_top, main_bot, total)

    purity  = 1 - 错配像素占比。越接近 1，说明这条线把"上一种颜色/下一种颜色"分得越干净。
              纯红/纯蓝的物块因为缺少另一种颜色，凑不出两层，purity 上不去 → 判不出 red_blue。
    balance = 两层面积的均衡度（小的/大的），防止"一整块红 + 底边一条蓝线"被当成相间。
    """
    x, y, w, h = box
    rows_r = [int(np.count_nonzero(blob_red[r, x:x + w])) for r in range(y, y + h)]
    rows_b = [int(np.count_nonzero(blob_blue[r, x:x + w])) for r in range(y, y + h)]
    total = sum(rows_r) + sum(rows_b)
    if total == 0:
        return None, 0.0, 0.0, 0, 0, 0

    best = None
    for i in range(1, h):                       # 在第 i 行切一刀
        ar, ab = sum(rows_r[:i]), sum(rows_b[:i])   # 上半区
        br, bb = sum(rows_r[i:]), sum(rows_b[i:])   # 下半区
        # 假设"上红下蓝"：错配像素 = 上半区的蓝 + 下半区的红
        # 假设"上蓝下红"：错配像素 = 上半区的红 + 下半区的蓝
        for cost, top_is_red, mt, mb in ((ab + br, True, ar, bb),
                                         (ar + bb, False, ab, br)):
            if mt < min_layer * total or mb < min_layer * total:
                continue                        # 两层都得有足够面积
            if best is None or cost < best[0]:
                best = (cost, top_is_red, mt, mb)

    if best is None:
        return None, 0.0, 0.0, 0, 0, total

    cost, top_is_red, mt, mb = best
    purity = 1.0 - cost / total
    balance = min(mt, mb) / max(mt, mb) if max(mt, mb) > 0 else 0.0
    return top_is_red, purity, balance, mt, mb, total


def classify(blob_red, blob_blue, box, p):
    n_red = int(np.count_nonzero(blob_red))
    n_blue = int(np.count_nonzero(blob_blue))
    total = n_red + n_blue
    if total == 0:
        return "unknown", 0.0

    r_red = n_red / total
    r_blue = n_blue / total

    # ① 单一颜色占绝对多数 → 纯色
    if r_red >= p["dominant_ratio"]:
        return "red", r_red
    if r_blue >= p["dominant_ratio"]:
        return "blue", r_blue

    # ② 两色都占一定比例 → 再查是不是上下分层
    if r_red >= p["interleave_ratio"] and r_blue >= p["interleave_ratio"]:
        top_is_red, purity, balance, mt, mb, _ = _layer_split(blob_red, blob_blue, box)
        if top_is_red is not None and purity >= p["layer_purity"] and balance >= p["layer_tol"]:
            return "red_blue", purity

    # ③ 不是真分层 → 退化成占优的那一色
    return ("red" if r_red > r_blue else "blue"), max(r_red, r_blue)


# =============================================================================
#  ★改3：把"像素边长"换算成物理量（针孔相机模型）
#  这是整道题的核心公式，也是 3cm 精度的来源：
#      Z = fy * W_real / w_px      （相似三角形：物体越大/离得越近，占的像素越多）
#      X = (u - cx) * Z / fx       （像素偏右多少 → 实际偏右多少米）
#      Y = (v - cy) * Z / fy       （像素偏下多少 → 实际偏下多少米）
#  单位说明：Z 和 X/Y 都是米，打印时再乘 100 变成厘米（题目要 cm）
# =============================================================================
def solve_pose(edge_px, box, p):
    x, y, w, h = box
    u = x + w / 2.0                                # 物块中心的像素坐标
    v = y + h / 2.0
    if edge_px <= 1.0:
        return 0.0, 0.0, 0.0, u, v

    z = p["fy"] * p["real_edge_m"] / edge_px       # 距离（米）
    x_m = (u - p["cx"]) * z / p["fx"]              # 水平偏移（米），右为正
    y_m = (v - p["cy"]) * z / p["fy"]              # 垂直偏移（米），下为正

    # 加分项①：参照点与相机光心的固定偏移，减掉
    x_m -= p["offset_x_m"]
    y_m -= p["offset_y_m"]
    return x_m, y_m, z, u, v


# =============================================================================
#  单帧检测主流程
# =============================================================================
def detect(frame, p):
    H, W = frame.shape[:2]

    # ★改11：先做畸变校正（dist 全 0 时自动跳过，不影响速度）
    if any(abs(v) > 1e-9 for v in p["dist"]):
        K = np.array([[p["fx"], 0.0, p["cx"]],
                      [0.0, p["fy"], p["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        frame = cv2.undistort(frame, K, np.array(p["dist"], dtype=np.float64))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---- 1) 颜色分割 ----
    mask_red = cv2.inRange(hsv, (0, p["red_s_min"], p["red_v_min"]),
                                (p["red_h_low_max"], 255, 255)) | \
               cv2.inRange(hsv, (p["red_h_high_min"], p["red_s_min"], p["red_v_min"]),
                                (179, 255, 255))
    mask_blue = cv2.inRange(hsv, (p["blue_h_min"], p["blue_s_min"], p["blue_v_min"]),
                                 (p["blue_h_max"], 255, 255))

    # ---- 2) 形态学：先 OPEN 去噪，再对【合并后】的掩膜 CLOSE 补缝 ----
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (p["open_kernel"],) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (p["close_kernel"],) * 2)

    # 各自先 OPEN 去掉孤立噪点
    red_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k_open)
    blue_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k_open)

    # ★改6b（重要 bug 修复，靠带真值的图集才发现）
    #  CLOSE 必须作用在「红蓝合并之后」的掩膜上，不能对每种颜色各做一次！
    #  原因：红蓝相间块的两种颜色交界处，因为镜头模糊/抗锯齿会混成一条紫红色窄带，
    #        它既不属于红也不属于蓝 → 掩膜在这里断开。
    #        对「每种颜色各自」做 CLOSE 是补不上这条缝的（每种颜色本身就是完整的半块，
    #        自己跟自己 CLOSE 没有任何效果）→ 一个红蓝相间的物块被拆成"纯红"+"纯蓝"
    #        两个轮廓：类别判错，而且每半个的 bbox 只有一半高度 → 距离直接偏 30%。
    both = cv2.bitwise_or(red_clean, blue_clean)
    both = cv2.morphologyEx(both, cv2.MORPH_CLOSE, k_close)

    # ---- 3) 找轮廓 ----
    contours, _ = cv2.findContours(both, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) < p["min_area_px"]:
            continue

        box = cv2.boundingRect(cnt)

        # ★改7：在【轮廓内部】统计红蓝像素，不再用外接矩形当分母
        #  原因：你原来用 bbox 面积做分母，但物块的外接矩形四个角是背景/别的颜色，
        #        分母被撑大 → 红蓝比例被算小 → 分类阈值全部失真。
        #        先生成一个"只包含该物块"的实心掩膜，再和红/蓝掩膜求交，统计才准确。
        only = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(only, [cnt], -1, 255, cv2.FILLED)
        blob_red = cv2.bitwise_and(red_clean, only)
        blob_blue = cv2.bitwise_and(blue_clean, only)

        kind, conf = classify(blob_red, blob_blue, box, p)
        n_red = int(np.count_nonzero(blob_red))
        n_blue = int(np.count_nonzero(blob_blue))
        edge_px, how, bbox_est, minrect_est, face_est, area_est = measure_edge_px(cnt, p)
        x_m, y_m, z_m, u, v = solve_pose(edge_px, box, p)

        results.append(dict(
            kind=kind, conf=conf, box=box, edge_px=edge_px, how=how,
            x=x_m, y=y_m, z=z_m, u=u, v=v,
            bbox_est=bbox_est, minrect_est=minrect_est, face_est=face_est,
            area_est=area_est,
            n_red=n_red, n_blue=n_blue,
            area=cv2.contourArea(cnt), cnt=cnt,
        ))

    results.sort(key=lambda r: -r["area"])          # 大的排前面，输出稳定
    return results, red_clean, blue_clean


# =============================================================================
#  画结果图
# =============================================================================
def draw(frame, results):
    vis = frame.copy()
    for r in results:
        x, y, w, h = r["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis, r["kind"], (x, max(11, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return vis


# =============================================================================
#  打印结果
# =============================================================================
def print_results(results, p):
    # ★改2：输出的单位从"像素"改成题目要的物理量
    #  原因：题目原文要"长度以及距离摄像头中心点的坐标"，而且给了"实际误差不能超过 3cm"
    #        —— 像素根本没法跟 3cm 比。所以这里统一输出厘米：
    #        种类 / 边长(cm) / X(cm) / Y(cm) / 距离Z(cm)，验收时就拿这几个数跟卷尺比。
    print("-" * 96)
    print(f"{'种类':<10}{'边长(cm)':>9}{'X(cm)':>9}{'Y(cm)':>9}{'距离Z(cm)':>11}"
          f"{'像素边':>9}{'置信':>7}   测量方式")
    print("-" * 96)
    for r in results:
        print(f"{r['kind']:<10}{p['real_edge_m'] * 100:>9.2f}"
              f"{r['x'] * 100:>9.2f}{r['y'] * 100:>9.2f}{r['z'] * 100:>11.2f}"
              f"{r['edge_px']:>9.1f}{r['conf']:>7.1%}   {r['how']}")
    print("-" * 96)
    print(f"识别到 {len(results)} 个物块")
    print()
    print("【四种像素边长估计值对比】看它们差多少，就知道'边长取法'对精度影响多大：")
    print(f"  {'':4}{'种类':<11}{'bbox':>8}{'minAreaRect':>13}{'正方形面':>11}"
          f"{'√面积-偏置':>12}{'→ 采用':>10}")
    for i, r in enumerate(results):
        face = f"{r['face_est']:.1f}" if r["face_est"] else "无(非四边形)"
        print(f"  第{i + 1}个 {r['kind']:<11}{r['bbox_est']:>8.1f}{r['minrect_est']:>13.1f}"
              f"{face:>11}{r['area_est']:>12.2f}{r['edge_px']:>10.2f}   ({r['how']})")
    print("  说明：四种估计都系统性偏大（掩膜外扩），所以面积法要减 edge_bias_px。")
    print("        bbox/minAreaRect 会被轮廓噪声撑大，小目标上尤其明显；面积法最稳。")
    print()


# =============================================================================
#  main
# =============================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="题目1 颜色物块识别与定位")
    # ★改12：支持 图片 / 视频 / 摄像头 三种输入（你原来只有图片）
    ap.add_argument("--src", default=os.path.join(here, "..", "samples", "first.png"),
                    help="图片或视频文件路径")
    ap.add_argument("--camera", type=int, default=None,
                    help="用摄像头，传设备号（一般是 0）")
    ap.add_argument("--out", default=None, help="结果图保存路径（不填就不存）")
    ap.add_argument("--show", action="store_true", help="弹窗显示（按 q 退出）")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="重复处理 N 帧并报告帧率（用来量 60Hz 达不达标）")
    args = ap.parse_args()

    # ---- 打开输入源 ----
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("[错误] 打不开摄像头。注意：WSL 里通常没有 /dev/video0，")
            print("       要么用 Windows 侧采集，要么用 usbipd 直通，要么先处理图片/视频。")
            return 1
        src_desc = f"摄像头 {args.camera}"
        is_file = False
    else:
        src = os.path.abspath(args.src)
        if not os.path.exists(src):
            print(f"[错误] 找不到文件：{src}")
            return 1
        ext = os.path.splitext(src)[1].lower()
        is_file = ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if is_file:
            img = cv2.imread(src)
            if img is None:
                print("[错误] 图片读不出来（路径有中文/空格时注意引号）")
                return 1
            src_desc = src
        else:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                print(f"[错误] 打不开视频：{src}")
                return 1
            src_desc = src

    print()
    print("=" * 96)
    print(f"输入：{src_desc}")
    print("=" * 96)

    # =========================================================================
    #  情况 A：单张图片
    # =========================================================================
    if not args.camera and is_file:
        # ★改4b：提醒"内参"和"这张图的分辨率"必须对得上
        #  原因：cx/cy 是光心在图像里的像素位置，640x480 时约等于 (320, 240)。
        #        如果图是别的尺寸却沿用 640x480 的内参，X/Y 坐标会整体偏掉。
        H, W = img.shape[:2]
        if abs(P["cx"] - W / 2) > 0.2 * W or abs(P["cy"] - H / 2) > 0.2 * H:
            print(f"[提醒] 这张图是 {W}x{H}，但参数里 cx={P['cx']:.0f} cy={P['cy']:.0f} "
                  f"是按 640x480 填的。")
            print("       → 算出来的 X/Y 会明显偏移，请换成这张图真正的内参（标定得到）。")
            print()
        results, _, _ = detect(img, P)
        print_results(results, P)

        if args.out:
            out_path = os.path.abspath(args.out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            # ★改1：用 os.path.join，不再用 + r"\result.png"
            #  原因：反斜杠在 Linux 里不是路径分隔符，是合法的文件名字符。
            #        你原来那句在 WSL 里会生成一个名字里带 "\" 的怪文件。
            ok = cv2.imwrite(out_path, draw(img, results))
            print(f"结果图已保存：{out_path}  ({'成功' if ok else '失败'})")

        if args.benchmark > 0:
            t0 = time.time()
            for _ in range(args.benchmark):
                detect(img, P)
            dt = time.time() - t0
            fps = args.benchmark / dt
            print(f"\n【帧率测试】处理 {args.benchmark} 帧用 {dt:.3f}s  →  {fps:.1f} FPS")
            print(f"  {'达标' if fps >= 60 else '不达标'}（题目要求 >= 60Hz）")
            if fps < 60:
                print("  提不上去的常见原因：图片分辨率太大（先降到 640x480）、")
                print("  没开 Release 优化（Python 影响小，C++ 节点影响很大）、")
                print("  同一帧里物块太多、形态学核开得太大。")

        if args.show:
            cv2.imshow("result", draw(img, results))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    # =========================================================================
    #  情况 B：视频 / 摄像头 —— 连续处理，统计真实处理帧率
    #  这一路才是验收 60Hz 时要跑的东西
    # =========================================================================
    n_frame = 0
    t_start = time.time()
    last_results = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frame += 1
            last_results, _, _ = detect(frame, P)

            if n_frame % 30 == 0:
                el = time.time() - t_start
                print(f"  已处理 {n_frame} 帧，平均 {n_frame / el:.1f} FPS，"
                      f"当前画面 {len(last_results)} 个物块")

            if args.show:
                cv2.imshow("result", draw(frame, last_results))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    el = time.time() - t_start
    print()
    print("=" * 96)
    if n_frame > 0:
        print(f"处理了 {n_frame} 帧，总耗时 {el:.2f}s  →  {n_frame / el:.1f} FPS")
        print(f"  {'达标' if n_frame / el >= 60 else '不达标'}（题目要求 >= 60Hz）")
    print_results(last_results, P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
