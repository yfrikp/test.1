"""
detector.py —— 题目1 检测算法的跨平台参考实现

设计目的：
  1) 与 C++ 版 (color_detector_node.cpp) 逐行对应，方便你两边对照学习；
  2) 不依赖 ROS2，在 Windows 上装了 Python + OpenCV 就能跑，
     让你在环境还没就绪时先把算法和阈值调通；
  3) 提供可复现的离线自测（配 make_test_images.py 生成的合成图）。

算法流程（与 C++ 完全一致）：
  BGR -> HSV -> 红(双区间)/蓝阈值分割 -> 形态学去噪 -> 找最大轮廓
      -> 轮廓内统计红蓝像素占比 -> 分类(纯色 or 相间+分层交叉验证)
      -> minAreaRect 取像素边长 -> 针孔模型算距离与三维坐标

重要约定：
  * 距离用 minAreaRect 的【长边】算。理由见 pixel_edge_for_distance 的注释。
  * 所有 print 输出都是英文 —— 中文在 GBK 终端下会乱码。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
#  参数
# =============================================================================
@dataclass
class DetectorParams:
    """所有可调参数。字段名与 config/params.yaml 一一对应。"""

    # ---- 红色 HSV 双区间（OpenCV 的 H 是 0~179）----
    red_h_low_max: int = 10        # 低段 H 上界：  0 ~ this
    red_h_high_min: int = 170      # 高段 H 下界：this ~ 179
    red_s_min: int = 90            # 饱和度下限：滤掉发白的红
    red_v_min: int = 50            # 亮度下限：滤掉发黑的红

    # ---- 蓝色 HSV ----
    blue_h_min: int = 95
    blue_h_max: int = 135
    blue_s_min: int = 90
    blue_v_min: int = 50

    # ---- 分类阈值 ----
    dominant_ratio: float = 0.80      # 单色占比 >= 此值 -> 纯色
    interleave_ratio: float = 0.18    # 相间时每色至少占比
    layer_split_ratio: float = 0.5    # 分层切分位置（0.5=对半切；别用 0.25，见 layer_check 注释）
    layer_size_tol: float = 0.45      # 上下层均衡度下限

    # ---- 预处理 ----
    blur_kernel: int = 5              # 必须是奇数
    morph_kernel: int = 5             # 必须是奇数
    min_area_px: int = 200            # 小于此面积当噪声

    # ---- 物理换算 ----
    real_edge_m: float = 0.05         # ★物块真实边长（米）—— 必须实测填写
    corner_offset_m: float = 0.0      # ★加分项：参照点偏移（米）

    # ---- 相机内参（无 camera_info 时的兜底）----
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0

    # ---- 额外 ----
    distance_mode: str = "pixel_height"   # "pixel_height" | "ground_plane"
    cam_height_m: float = 0.35
    cam_pitch_deg: float = 30.0

    def to_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


# =============================================================================
#  结果
# =============================================================================
@dataclass
class Detection:
    block_class: str = "unknown"      # "red" | "blue" | "red_blue" | "unknown"
    edge_length_m: float = 0.0
    edge_length_px: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    u_px: float = 0.0
    v_px: float = 0.0
    confidence: float = 0.0
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h
    ratio_red: float = 0.0
    ratio_blue: float = 0.0
    layer_ok: bool = False
    debug: Dict[str, float] = field(default_factory=dict)


# =============================================================================
#  工具
# =============================================================================
def _odd(k: int) -> int:
    """把核大小强制成 >=1 的奇数（形态学和模糊都要求奇数）"""
    k = int(k)
    if k < 1:
        k = 1
    return k if k % 2 == 1 else k + 1


def build_masks(hsv: np.ndarray, p: DetectorParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成红色掩膜和蓝色掩膜。

    为什么红色要用两段 H 区间：
      红色在 HSV 色环上跨过 0 度，OpenCV 的 H 是 0~179，
      所以纯红表现为 0~10 和 170~179 两段，必须合并，否则会漏检一半。
    """
    hsv_b = cv2.GaussianBlur(hsv, (_odd(p.blur_kernel),) * 2, 0)

    lower_red_1 = np.array([0, p.red_s_min, p.red_v_min], dtype=np.uint8)
    upper_red_1 = np.array([p.red_h_low_max, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([p.red_h_high_min, p.red_s_min, p.red_v_min], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv_b, lower_red_1, upper_red_1),
        cv2.inRange(hsv_b, lower_red_2, upper_red_2),
    )

    lower_blue = np.array([p.blue_h_min, p.blue_s_min, p.blue_v_min], dtype=np.uint8)
    upper_blue = np.array([p.blue_h_max, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv_b, lower_blue, upper_blue)

    mk = _odd(p.morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    for m in (mask_red, mask_blue):
        cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, dst=m)    # 去小白点
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, dst=m)   # 补内部小洞

    return mask_red, mask_blue


def pixel_edge_for_distance(rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> float:
    """
    从 minAreaRect 结果里取"用于算距离的像素边长"。

    ★ 这里取【长边】，不是短边。原因：
      物块是立方体，正对相机时投影是正方形，两边相等；
      侧视时投影变成矩形，长边仍然对应立方体的棱长（透视缩短只影响短边）。
      所以长边更接近真实棱长的投影 —— 用短边会把距离算得偏大。

    minAreaRect 返回 ((cx,cy), (w,h), angle)，w/h 无序，所以取 max。
    """
    (_c, (w, h), _a) = rect
    return float(max(w, h))


def classify(
    n_red: float, n_blue: float, p: DetectorParams,
    mask_red_blob: np.ndarray, mask_blue_blob: np.ndarray, box: Tuple[int, int, int, int],
) -> Tuple[str, float, float, float, bool]:
    """
    分类：返回 (类别, 置信度, 红占比, 蓝占比, 分层是否通过)

    顺序很重要：先排除"单色占绝对多数"，再判相间，最后用分层几何证据交叉验证。
    """
    total = n_red + n_blue
    if total <= 0:
        return "unknown", 0.0, 0.0, 0.0, False

    r_red = n_red / total
    r_blue = n_blue / total

    if r_red >= p.dominant_ratio and r_blue < p.interleave_ratio:
        return "red", r_red, r_red, r_blue, False
    if r_blue >= p.dominant_ratio and r_red < p.interleave_ratio:
        return "blue", r_blue, r_red, r_blue, False
    if r_red >= p.interleave_ratio and r_blue >= p.interleave_ratio:
        conf = min(r_red, r_blue) * 2.0
        ok = layer_check(mask_red_blob, mask_blue_blob, box, p)
        if ok:
            return "red_blue", conf, r_red, r_blue, True
        # 分层不通过 -> 退化占优的那一色（避免"红块+蓝色反光"被误判成相间）
        if r_red >= r_blue:
            return "red", r_red, r_red, r_blue, False
        return "blue", r_blue, r_red, r_blue, False

    return "unknown", 0.0, r_red, r_blue, False


def layer_check(
    mask_red: np.ndarray, mask_blue: np.ndarray, box: Tuple[int, int, int, int],
    p: DetectorParams,
) -> bool:
    """
    上下分层检查：确认"红蓝相间"真的是分层的，而不是"一整块某色 + 边缘杂色"。

    判据：
      1) 上半以某色为主、下半以另一色为主（"上红下蓝" 或 "上蓝下红" 都接受，
         因为相机可能拍到正面也可能拍到背面）
      2) 上层的主色层 与 下层的主色层，规模要均衡（min/max >= layer_size_tol），
         否则说明只是一小块杂色附在大块上
    """
    x, y, w, h = box
    if h < 8 or w < 8:
        return False

    y0 = max(0, y)
    y1 = min(mask_red.shape[0], y + h)
    x0 = max(0, x)
    x1 = min(mask_red.shape[1], x + w)
    split = y0 + int((y1 - y0) * p.layer_split_ratio)
    if split <= y0 or split >= y1:
        return False

    up_r = int(np.count_nonzero(mask_red[y0:split, x0:x1]))
    up_b = int(np.count_nonzero(mask_blue[y0:split, x0:x1]))
    lo_r = int(np.count_nonzero(mask_red[split:y1, x0:x1]))
    lo_b = int(np.count_nonzero(mask_blue[split:y1, x0:x1]))

    red_top = (up_r > up_b) and (lo_b > lo_r)
    blue_top = (up_b > up_r) and (lo_r > lo_b)
    layered = red_top or blue_top

    # 均衡度：比较【上层的主色层】和【下层的主色层】，不要拿"上下两层总面积"比。
    #
    # ★ 这里修了一个会让红蓝相间【永远判不出来】的 bug：
    #   两层高度由 layer_split_ratio 决定。早期默认 0.25（上 25% / 下 75%），
    #   两层总面积之比恒为 0.333，而 layer_size_tol 是 0.45 —— 永远达不到，
    #   layer_check 永远返回 False，红蓝相间每次都被退回成占优的那一色。
    #   改成比较两个颜色层自身的规模，才对应"两层大小相当"的本意。
    dom_up = up_r if red_top else up_b
    dom_low = lo_b if red_top else lo_r
    if dom_up + dom_low <= 0:
        return False
    balance = min(dom_up, dom_low) / max(dom_up, dom_low)
    return bool(layered and balance >= p.layer_size_tol)


# =============================================================================
#  主检测函数
# =============================================================================
def detect(bgr: np.ndarray, p: DetectorParams) -> Detection:
    """对一张 BGR 图像做检测，返回 Detection"""
    out = Detection()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_red, mask_blue = build_masks(hsv, p)
    mask_any = cv2.bitwise_or(mask_red, mask_blue)

    # ★ 补上"红蓝交界处的缝隙"（不补的话红蓝相间会被判成单色）：
    #   build_masks() 里对 HSV 做了高斯模糊，红蓝相接处的 H 从 0 平滑过渡到 120，
    #   中间几个像素既不属于红也不属于蓝，mask_any 正好在这里断开 ->
    #   findContours 得到两块 -> "取面积最大的"只拿到其中一块（实测被判成 blue）。
    #   闭运算可以把窄缝连起来，外轮廓尺寸不变（实测 244x244）。
    _gap = max(7, _odd(p.morph_kernel) * 2 + 1)
    mask_any = cv2.morphologyEx(
        mask_any, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_gap, _gap)),
    )

    contours, _ = cv2.findContours(mask_any, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out

    # 选面积最大的轮廓（题目一次只需识别一个物块）
    areas = [cv2.contourArea(c) for c in contours]
    best = int(np.argmax(areas))
    if areas[best] < p.min_area_px:
        return out

    cnt = contours[best]
    box = cv2.boundingRect(cnt)

    # ---- 统计轮廓内的红蓝像素 ----
    # 先把该轮廓单独画到空白掩膜上，再与红/蓝掩膜求交，
    # 这样统计严格限定在这个物块内部，不受画面里其他红色物体影响。
    only = np.zeros(mask_any.shape, dtype=np.uint8)
    cv2.drawContours(only, [cnt], -1, 255, cv2.FILLED)
    blob_red = cv2.bitwise_and(mask_red, only)
    blob_blue = cv2.bitwise_and(mask_blue, only)

    n_red = float(np.count_nonzero(blob_red))
    n_blue = float(np.count_nonzero(blob_blue))
    if n_red + n_blue < 1:
        return out

    cls, conf, r_red, r_blue, layer_ok = classify(
        n_red, n_blue, p, blob_red, blob_blue, box)

    # ---- 像素边长 ----
    rect = cv2.minAreaRect(cnt)
    px_edge = pixel_edge_for_distance(rect)

    # ---- 距离与三维坐标（针孔模型）----
    z = x = y = 0.0
    if px_edge > 1.0:
        if p.distance_mode == "ground_plane":
            z = ground_plane_distance(box, p)
        else:
            # Z = fy * 真实边长 / 像素边长（相似三角形）
            z = p.fy * p.real_edge_m / px_edge
        uc = box[0] + box[2] * 0.5
        vc = box[1] + box[3] * 0.5
        # 反投影：(u-cx) 是"偏右多少像素"，乘 Z/fx 变成"偏右多少米"
        x = (uc - p.cx) * z / p.fx
        y = (vc - p.cy) * z / p.fy

    # ---- 加分项：参照点偏移补偿 ----
    # P_ref = P_cam - t_offset（相机坐标系下平移）
    if abs(p.corner_offset_m) > 1e-9:
        x -= p.corner_offset_m

    out.block_class = cls
    out.edge_length_m = p.real_edge_m if px_edge > 1.0 else 0.0
    out.edge_length_px = px_edge
    out.x_m, out.y_m, out.z_m = x, y, z
    out.u_px = box[0] + box[2] * 0.5
    out.v_px = box[1] + box[3] * 0.5
    out.confidence = conf
    out.box = tuple(int(v) for v in box)  # type: ignore[assignment]
    out.ratio_red, out.ratio_blue = r_red, r_blue
    out.layer_ok = layer_ok
    out.debug = {
        "n_red": n_red, "n_blue": n_blue, "area": areas[best],
        "rect_w": rect[1][0], "rect_h": rect[1][1],
    }
    return out


def ground_plane_distance(box: Tuple[int, int, int, int], p: DetectorParams) -> float:
    """
    地面平面模式：物块放在桌面上、相机固定俯视时用。

    原理：相机俯角 theta、高度 h，物块底边垂直像素 v。
          与光轴夹角 alpha = atan((v - cy) / fy)
          地面距离 Z = h * tan(theta + alpha)

    比"像素高度"稳，因为像素高度受物块朝向影响很大，而底边位置只取决于它在桌面上的位置。
    """
    v_bottom = box[1] + box[3]
    alpha = math.atan((v_bottom - p.cy) / max(1.0, p.fy))
    theta = math.radians(p.cam_pitch_deg)
    z = p.cam_height_m * math.tan(theta + alpha)
    return z if math.isfinite(z) and z > 0 else 0.0


# =============================================================================
#  可视化（供 tune_hsv.py / selftest.py 复用）
# =============================================================================
def draw_result(bgr: np.ndarray, det: Detection, params: DetectorParams) -> np.ndarray:
    """
    在图上画出检测结果。文字一律用英文 —— OpenCV 的 putText 不支持中文，
    写中文会变成一串问号（这也是为什么整个项目的终端/图上文字都用英文）。
    """
    canvas = bgr.copy()
    h, w = canvas.shape[:2]

    # 画面中心十字：直观看出"物块离相机中心偏了多少"
    cv2.drawMarker(canvas, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)

    x, y, bw, bh = det.box
    if bw > 0 and bh > 0:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        lines = [
            f"{det.block_class}  conf={det.confidence:.2f}",
            f"z={det.z_m:.4f}m  x={det.x_m:+.4f}m  y={det.y_m:+.4f}m",
            f"edge={det.edge_length_m:.4f}m ({det.edge_length_px:.1f}px)",
            f"R={det.ratio_red:.2f} B={det.ratio_blue:.2f} layer={det.layer_ok}",
        ]
        ty = max(18, y - 8 - 16 * (len(lines) - 1))
        for i, s in enumerate(lines):
            org = (x, ty + 16 * i)
            cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(canvas, "no target", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return canvas


DEFAULT_PARAMS = DetectorParams()
