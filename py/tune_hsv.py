"""
tune_hsv.py —— 交互式 HSV 阈值调参工具（带滑条）

用途：题目1 里最难的一步是"把阈值调到三种物块都能稳定识别"。
     凭感觉改数字效率极低，这个工具让你拖滑条实时看掩膜效果。

用法：
    python tune_hsv.py --image samples/synth_red_d0500mm_xp0000_yp0000_L100.png
    python tune_hsv.py --image photo.jpg --edge 0.05     # 顺便看距离估算

操作：
    拖动上排滑条  -> 调 HSV 阈值 / 形态学 / 分类阈值
    按 s          -> 把当前参数保存成 params_tuned.yaml
    按 d          -> 在"显示红蓝掩膜"和"只显示结果"之间切换
    按 n / p      -> 切到下一张 / 上一张图（需要 --image 传目录或用 --glob）
    按 q 或 ESC   -> 退出

窗口里的文字全是英文 —— OpenCV 的 putText 不支持中文，写中文会变成问号。
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import sys
from typing import List

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DetectorParams, build_masks, detect, draw_result  # noqa: E402

WIN_CTRL = "controls"
WIN_MAIN = "result  (s=save  d=toggle mask  n/p=next/prev  q=quit)"


# =============================================================================
#  滑条定义
#  每项：(标签, 参数名, 最小值, 最大值)
# =============================================================================
SLIDERS = [
    # ---- 红色双区间 ----
    ("red H low  max (0..this)",   "red_h_low_max",     0,  40),
    ("red H high min (this..179)", "red_h_high_min",  140, 179),
    ("red S min",                  "red_s_min",         0, 255),
    ("red V min",                  "red_v_min",         0, 255),
    # ---- 蓝色 ----
    ("blue H min",                 "blue_h_min",        0, 179),
    ("blue H max",                 "blue_h_max",        0, 179),
    ("blue S min",                 "blue_s_min",        0, 255),
    ("blue V min",                 "blue_v_min",        0, 255),
    # ---- 预处理 ----
    ("blur kernel (odd)",          "blur_kernel",       1,  31),
    ("morph kernel (odd)",         "morph_kernel",      1,  31),
    ("min area px",                "min_area_px",       0, 5000),
    # ---- 分类 ----
    ("dominant ratio x100",        "dominant_ratio_x100",   0, 100),
    ("interleave ratio x100",      "interleave_ratio_x100", 0, 100),
    ("layer split x100",           "layer_split_ratio_x100", 5, 95),
    ("layer size tol x100",        "layer_size_tol_x100",   0, 100),
]


def create_sliders(p: DetectorParams) -> None:
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 560, 480)
    for label, name, lo, hi in SLIDERS:
        init = getattr(p, name, None)
        if init is None:
            # 比率类参数在 Params 里存的是 0~1，滑条上是 0~100
            base = name.replace("_x100", "")
            init = int(round(float(getattr(p, base)) * 100))
        init = int(max(lo, min(hi, init)))
        cv2.createTrackbar(label, WIN_CTRL, init, hi, lambda _v: None)


def read_sliders(p: DetectorParams) -> None:
    for label, name, _lo, _hi in SLIDERS:
        v = cv2.getTrackbarPos(label, WIN_CTRL)
        if name.endswith("_x100"):
            setattr(p, name.replace("_x100", ""), v / 100.0)
        else:
            setattr(p, name, int(v))


def save_params(p: DetectorParams, path: str) -> None:
    """保存成与 config/params.yaml 相同结构的文件"""
    lines = [
        "# 由 tune_hsv.py 自动保存 —— 可直接复制到 config/params.yaml 的 ros__parameters 下",
        "color_detector:",
        "  ros__parameters:",
        f"    # ★ 物块真实边长，必须实测（这里保留你命令行传的值）",
        f"    real_edge_m: {p.real_edge_m}",
        f"    corner_offset_m: {p.corner_offset_m}",
        f'    distance_mode: "{p.distance_mode}"',
        f"    cam_height_m: {p.cam_height_m}",
        f"    cam_pitch_deg: {p.cam_pitch_deg}",
        "",
        f"    red_h_low_max: {p.red_h_low_max}",
        f"    red_h_high_min: {p.red_h_high_min}",
        f"    red_s_min: {p.red_s_min}",
        f"    red_v_min: {p.red_v_min}",
        "",
        f"    blue_h_min: {p.blue_h_min}",
        f"    blue_h_max: {p.blue_h_max}",
        f"    blue_s_min: {p.blue_s_min}",
        f"    blue_v_min: {p.blue_v_min}",
        "",
        f"    dominant_ratio: {p.dominant_ratio:.2f}",
        f"    interleave_ratio: {p.interleave_ratio:.2f}",
        f"    layer_split_ratio: {p.layer_split_ratio:.2f}",
        f"    layer_size_tol: {p.layer_size_tol:.2f}",
        "",
        f"    morph_kernel: {p.morph_kernel}",
        f"    blur_kernel: {p.blur_kernel}",
        f"    min_area_px: {p.min_area_px}",
        "",
        '    image_path: ""',
        "    republish_hz: 60.0",
        "    show_image: false",
        "",
        f"    fallback_fx: {p.fx}",
        f"    fallback_fy: {p.fy}",
        f"    fallback_cx: {p.cx}",
        f"    fallback_cy: {p.cy}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def gather_images(image_arg: str) -> List[str]:
    if os.path.isdir(image_arg):
        out: List[str] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            out += sorted(globmod.glob(os.path.join(image_arg, ext)))
        return out
    if any(ch in image_arg for ch in "*?[]"):
        return sorted(globmod.glob(image_arg))
    return [image_arg] if os.path.isfile(image_arg) else []


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive HSV threshold tuner")
    ap.add_argument("--image", required=True, help="image file, directory, or glob")
    ap.add_argument("--edge", type=float, default=0.05, help="real_edge_m for distance estimate")
    ap.add_argument("--fx", type=float, default=600.0)
    ap.add_argument("--fy", type=float, default=600.0)
    ap.add_argument("--out", default="params_tuned.yaml", help="where to save with 's'")
    args = ap.parse_args()

    files = gather_images(args.image)
    if not files:
        print(f"[FAIL] no image found for: {args.image}")
        print("       generate some first:  python make_test_images.py --out samples")
        return 2

    p = DetectorParams()
    p.real_edge_m = args.edge
    p.fx = args.fx
    p.fy = args.fy
    # cx/cy 按图像尺寸设置（后面每张图都会重算）
    idx = 0
    show_mask = True

    img = cv2.imread(files[idx], cv2.IMREAD_COLOR)
    if img is None:
        print(f"[FAIL] cannot read {files[idx]}")
        return 2
    p.cx, p.cy = img.shape[1] / 2.0, img.shape[0] / 2.0

    create_sliders(p)
    print("[OK] tuner started.")
    print(f"     image: {files[idx]}")
    print("     keys : s=save  d=toggle masks  n/p=next/prev  q=quit")

    while True:
        # 从滑条读回参数
        read_sliders(p)

        det = detect(img, p)

        main_view = draw_result(img, det, p)

        if show_mask:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mr, mb = build_masks(hsv, p)
            # 把两种掩膜并排贴到主视图下方，便于判断"漏检/多检"
            h, w = img.shape[:2]
            masks = np.zeros((h, w * 2, 3), dtype=np.uint8)
            masks[:, :w] = cv2.cvtColor(mr, cv2.COLOR_GRAY2BGR)
            masks[:, w:] = cv2.cvtColor(mb, cv2.COLOR_GRAY2BGR)
            cv2.putText(masks, "RED mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(masks, "BLUE mask", (w + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            scale = w / masks.shape[1]
            masks = cv2.resize(masks, (int(masks.shape[1] * scale), int(masks.shape[0] * scale)))
            main_view = np.vstack([main_view, masks])

        cv2.imshow(WIN_MAIN, main_view)

        # 打印一行关键数据，方便边调边看
        print("\rclass={:<9} conf={:.2f} z={:.4f}m x={:+.4f}m y={:+.4f}m area={:>7.0f}   "
              .format(det.block_class, det.confidence, det.z_m, det.x_m, det.y_m,
                      det.debug.get("area", 0.0)), end="", flush=True)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            save_params(p, args.out)
            print("")
            print(f"[OK] saved -> {os.path.abspath(args.out)}")
            print("     copy the ros__parameters block into config/params.yaml")
        elif key == ord("d"):
            show_mask = not show_mask
        elif key in (ord("n"), ord("p")):
            step = 1 if key == ord("n") else -1
            idx = (idx + step) % len(files)
            ni = cv2.imread(files[idx], cv2.IMREAD_COLOR)
            if ni is not None:
                img = ni
                p.cx, p.cy = img.shape[1] / 2.0, img.shape[0] / 2.0
                print("")
                print(f"[OK] image {idx+1}/{len(files)}: {files[idx]}")

    cv2.destroyAllWindows()
    print("")
    print("[OK] tuner closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
