"""
make_test_images.py —— 生成带"标准答案"的合成测试图

为什么需要这个：
  题目要求「实际误差不能超过 3cm」。要验证精度，你必须知道真实值是多少。
  但你现在既没有相机也没有物块，更没法精确摆放。
  所以这里用代码合成图片：物块的位置、尺寸、距离都是我自己设定的，
  于是"标准答案"是精确已知的，可以直接算误差。

  这让你在【环境还没装好、手上什么都没有】的情况下，就能：
    * 验证算法流程能否跑通
    * 验证距离/坐标换算公式是否正确（误差应该是 0，因为它由同一套公式生成）
    * 验证三种物块能否被正确分类
    * 调 HSV 阈值在噪声和光照变化下的鲁棒性

用法：
    python make_test_images.py --out samples --cam-w 640 --cam-h 480 --fx 600 --fy 600
    python make_test_images.py --out samples --distances 0.30 0.50 1.00 --noise 6

输出：
    samples/synth_<类别>_d<距离>_x<偏移>_y<偏移>_<光照>.png    图片
    samples/synth_<...>.txt                                    该图的精确真值
    samples/ground_truth.csv                                   所有图的真值汇总表

注意：所有 print 都是英文（中文在 GBK 终端会乱码）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np


# =============================================================================
#  物块绘制
# =============================================================================
def make_block_patch(edge_px: int, block_class: str) -> np.ndarray:
    """
    生成一个边长为 edge_px 的物块贴图（BGR）。

    颜色取值注意：必须落在 detector.py 的 HSV 阈值范围内，否则会被漏检。
    这里用饱和的红 (0,0,255) 和蓝 (255,0,0) 及其在 HSV 下的等价色，
    但在后面会用光照系数整体缩放亮度，所以亮度不能太接近下限。
    """
    e = max(4, int(edge_px))
    patch = np.zeros((e, e, 3), dtype=np.uint8)

    # 纯红 / 纯蓝：BGR
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)

    if block_class == "red":
        patch[:] = RED
    elif block_class == "blue":
        patch[:] = BLUE
    elif block_class == "red_blue":
        # 上下分层：默认上红下蓝。生成时也会造一批"上蓝下红"来测对称性。
        split = e // 2
        patch[:split] = RED
        patch[split:] = BLUE

    return patch


def draw_block(img: np.ndarray, center_uv, edge_px: int, block_class: str,
               flip_layers: bool = False) -> int:
    """
    把物块画到图上（原地修改）。物块是轴对齐正方形，便于精确计算像素边长。

    ★改：返回**实际画出来的边长（整数像素）**。
      原因：像素只能是整数，ideal 的 edge_px（比如 37.5）会被 int(round()) 变成 38。
      如果真值里还记 37.5，评测时就会凭空多出一个"系统性误差"
      ——实测在 80cm 处虚报了 1.59cm，把真正的算法误差（0.26px）完全淹没。
      所以真值必须记录"实际画了多少像素"，并据此反算等效距离。
    """
    h, w = img.shape[:2]
    e = max(4, int(round(edge_px)))
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    x0 = max(0, u - e // 2)
    y0 = max(0, v - e // 2)
    x1 = min(w, x0 + e)
    y1 = min(h, y0 + e)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0

    patch = make_block_patch(e, block_class)
    if flip_layers and block_class == "red_blue":
        patch = patch[::-1].copy()      # 上下翻转 -> 上蓝下红
    # 裁剪到画布范围内
    patch = patch[: y1 - y0, : x1 - x0]
    img[y0:y1, x0:x1] = patch
    return min(x1 - x0, y1 - y0)        # 实际落笔的边长


# =============================================================================
#  合成一张图
# =============================================================================
def project(X: float, Y: float, Z: float, fx: float, fy: float, cx: float, cy: float):
    """
    针孔模型投影：相机坐标系下的三维点 -> 像素坐标。
    相机坐标系约定：x 右、y 下、z 前（与 ROS2 光学坐标系一致）。
    """
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v


def make_one(
    out_dir: str,
    block_class: str,
    distance: float,
    dx: float,
    dy: float,
    real_edge: float,
    fx: float, fy: float, cx: float, cy: float,
    img_w: int, img_h: int,
    light: float,
    noise_sigma: float,
    blur_ksize: int,
    bg: float,
    flip_layers: bool,
    jpeg_quality: int,
) -> dict:
    """合成一张图并写出图片 + 真值，返回真值 dict"""

    # ---- 背景：中灰 + 轻微渐变（模拟桌面的不均匀照明）----
    yy = np.linspace(0, 1, img_h, dtype=np.float32).reshape(-1, 1)
    base = (bg * (0.85 + 0.3 * yy))
    # 只做垂直渐变会得到 (img_h, 1) 的形状，dstack 之后图片只有 1 像素宽，
    # 物块根本画不出来（面积恒 < min_area_px）——必须补上水平方向。
    base = np.repeat(base, img_w, axis=1)
    img = np.dstack([base, base, base]).astype(np.float32)

    # ---- 计算像素边长并画物块 ----
    # 相似三角形：edge_px / fx = real_edge / Z   =>   edge_px = fx * real_edge / Z
    edge_px = fx * real_edge / distance
    u, v = project(dx, dy, distance, fx, fy, cx, cy)
    edge_px_drawn = draw_block(img, (u, v), edge_px, block_class, flip_layers)

    # ★ 真值要用"实际画出来的像素"反算等效距离。
    #   像素只能取整：ideal 37.5px 实际画成 38px，那么这张图**物理上**对应的距离
    #   就是 fx*real_edge/38，而不是 0.80m。用 ideal 当真值 = 让被测算法背黑锅。
    distance_eff = (fx * real_edge / edge_px_drawn) if edge_px_drawn else 0.0

    # ---- 整体光照系数（模拟明暗）----
    img = img * light

    # ---- 高斯模糊（模拟镜头失焦 / 抗锯齿）----
    if blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # ---- 加噪声（模拟传感器噪声）----
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, img.shape).astype(np.float32)
        img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)

    # ---- 文件名（把参数编码进去，便于人眼核对）----
    def tag(x: float) -> str:
        s = f"{x:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")
        return s
    name = (f"synth_{block_class}_d{int(round(distance*1000)):04d}mm"
            f"_x{tag(dx)}_y{tag(dy)}_L{int(light*100):03d}")
    if flip_layers:
        name += "_flip"
    png = os.path.join(out_dir, name + ".png")

    if jpeg_quality > 0:
        # 先编码成 JPEG 再解码，模拟真实照片的压缩损失
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imwrite(png, img)

    truth = {
        "file": os.path.basename(png),
        "block_class": block_class,
        "distance_m": round(distance, 6),
        # ★ distance_eff_m = 按"实际画出的像素"反算的等效距离，评测应该用它
        "distance_eff_m": round(distance_eff, 6),
        "edge_px_drawn": edge_px_drawn,
        "x_m": round(dx, 6),
        "y_m": round(dy, 6),
        "real_edge_m": real_edge,
        "edge_px": round(edge_px, 3),
        "u_px": round(u, 3),
        "v_px": round(v, 3),
        "light": light,
        "noise_sigma": noise_sigma,
        "jpeg_quality": jpeg_quality,
        "flip_layers": int(flip_layers),
    }

    # 同时写一份人可读的 txt
    with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
        for k, val in truth.items():
            f.write(f"{k}: {val}\n")

    return truth


# =============================================================================
#  主流程
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate synthetic test images with exact ground truth for vision_task1")
    ap.add_argument("--out", default="samples", help="output directory (default: samples)")
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--cam-h", type=int, default=480)
    ap.add_argument("--fx", type=float, default=600.0)
    ap.add_argument("--fy", type=float, default=600.0)
    ap.add_argument("--real-edge", type=float, default=0.05, help="block edge length in meters")
    ap.add_argument("--distances", type=float, nargs="+", default=[0.30, 0.50, 0.80, 1.20])
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.04, -0.04],
                    help="x/y offsets in meters (block center relative to camera axis)")
    ap.add_argument("--lights", type=float, nargs="+", default=[1.0, 0.7, 1.3],
                    help="brightness multipliers")
    ap.add_argument("--noise", type=float, default=4.0, help="gaussian noise sigma")
    ap.add_argument("--blur", type=int, default=3, help="gaussian blur kernel (odd)")
    ap.add_argument("--background", type=float, default=110.0, help="background gray level")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="0 = keep PNG lossless")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cx = args.cam_w / 2.0
    cy = args.cam_h / 2.0

    rows = []
    # 三种类别 × 多个距离 × 多个偏移 × 多个光照
    for block_class in ("red", "blue", "red_blue"):
        for d in args.distances:
            for dx in args.offsets:
                for dy in args.offsets:
                    for L in args.lights:
                        # 只对部分组合翻转分层，控制图片总数
                        flip = (block_class == "red_blue" and dx < 0 and L == args.lights[0])
                        t = make_one(
                            args.out, block_class, d, dx, dy, args.real_edge,
                            args.fx, args.fy, cx, cy, args.cam_w, args.cam_h,
                            L, args.noise, args.blur, args.background, flip,
                            args.jpeg_quality,
                        )
                        rows.append(t)

    csv_path = os.path.join(args.out, "ground_truth.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] generated {len(rows)} images into: {os.path.abspath(args.out)}")
    print(f"[OK] ground truth csv: {os.path.abspath(csv_path)}")
    print("")
    print("next step - run the offline self test:")
    print(f"  python selftest.py --samples {args.out}")
    print("")
    print("or tune HSV thresholds interactively:")
    print(f"  python tune_hsv.py --image {os.path.join(args.out, rows[0]['file'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
