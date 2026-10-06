from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import cv2
import numpy as np

def make_block_patch(edge_px: int, block_class: str) -> np.ndarray:
    e = max(4, int(edge_px))
    patch = np.zeros((e, e, 3), dtype=np.uint8)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    if block_class == 'red':
        patch[:] = RED
    elif block_class == 'blue':
        patch[:] = BLUE
    elif block_class == 'red_blue':
        split = e // 2
        patch[:split] = RED
        patch[split:] = BLUE
    return patch

def draw_block(img: np.ndarray, center_uv, edge_px: int, block_class: str, flip_layers: bool=False) -> int:
    (h, w) = img.shape[:2]
    e = max(4, int(round(edge_px)))
    (u, v) = (int(round(center_uv[0])), int(round(center_uv[1])))
    x0 = max(0, u - e // 2)
    y0 = max(0, v - e // 2)
    x1 = min(w, x0 + e)
    y1 = min(h, y0 + e)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0
    patch = make_block_patch(e, block_class)
    if flip_layers and block_class == 'red_blue':
        patch = patch[::-1].copy()
    patch = patch[:y1 - y0, :x1 - x0]
    img[y0:y1, x0:x1] = patch
    return min(x1 - x0, y1 - y0)

def project(X: float, Y: float, Z: float, fx: float, fy: float, cx: float, cy: float):
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return (u, v)

def make_one(out_dir: str, block_class: str, distance: float, dx: float, dy: float, real_edge: float, fx: float, fy: float, cx: float, cy: float, img_w: int, img_h: int, light: float, noise_sigma: float, blur_ksize: int, bg: float, flip_layers: bool, jpeg_quality: int) -> dict:
    yy = np.linspace(0, 1, img_h, dtype=np.float32).reshape(-1, 1)
    base = bg * (0.85 + 0.3 * yy)
    base = np.repeat(base, img_w, axis=1)
    img = np.dstack([base, base, base]).astype(np.float32)
    edge_px = fx * real_edge / distance
    (u, v) = project(dx, dy, distance, fx, fy, cx, cy)
    edge_px_drawn = draw_block(img, (u, v), edge_px, block_class, flip_layers)
    distance_eff = fx * real_edge / edge_px_drawn if edge_px_drawn else 0.0
    img = img * light
    if blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, img.shape).astype(np.float32)
        img = img + noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    def tag(x: float) -> str:
        s = f'{x:+.3f}'.replace('+', 'p').replace('-', 'm').replace('.', '')
        return s
    name = f'synth_{block_class}_d{int(round(distance * 1000)):04d}mm_x{tag(dx)}_y{tag(dy)}_L{int(light * 100):03d}'
    if flip_layers:
        name += '_flip'
    png = os.path.join(out_dir, name + '.png')
    if jpeg_quality > 0:
        (ok, buf) = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imwrite(png, img)
    truth = {'file': os.path.basename(png), 'block_class': block_class, 'distance_m': round(distance, 6), 'distance_eff_m': round(distance_eff, 6), 'edge_px_drawn': edge_px_drawn, 'x_m': round(dx, 6), 'y_m': round(dy, 6), 'real_edge_m': real_edge, 'edge_px': round(edge_px, 3), 'u_px': round(u, 3), 'v_px': round(v, 3), 'light': light, 'noise_sigma': noise_sigma, 'jpeg_quality': jpeg_quality, 'flip_layers': int(flip_layers)}
    with open(os.path.join(out_dir, name + '.txt'), 'w', encoding='utf-8') as f:
        for (k, val) in truth.items():
            f.write(f'{k}: {val}\n')
    return truth

def main() -> int:
    ap = argparse.ArgumentParser(description='Generate synthetic test images with exact ground truth for vision_task1')
    ap.add_argument('--out', default='samples', help='output directory (default: samples)')
    ap.add_argument('--cam-w', type=int, default=640)
    ap.add_argument('--cam-h', type=int, default=480)
    ap.add_argument('--fx', type=float, default=600.0)
    ap.add_argument('--fy', type=float, default=600.0)
    ap.add_argument('--real-edge', type=float, default=0.05, help='block edge length in meters')
    ap.add_argument('--distances', type=float, nargs='+', default=[0.3, 0.5, 0.8, 1.2])
    ap.add_argument('--offsets', type=float, nargs='+', default=[0.0, 0.04, -0.04], help='x/y offsets in meters (block center relative to camera axis)')
    ap.add_argument('--lights', type=float, nargs='+', default=[1.0, 0.7, 1.3], help='brightness multipliers')
    ap.add_argument('--noise', type=float, default=4.0, help='gaussian noise sigma')
    ap.add_argument('--blur', type=int, default=3, help='gaussian blur kernel (odd)')
    ap.add_argument('--background', type=float, default=110.0, help='background gray level')
    ap.add_argument('--jpeg-quality', type=int, default=92, help='0 = keep PNG lossless')
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cx = args.cam_w / 2.0
    cy = args.cam_h / 2.0
    rows = []
    for block_class in ('red', 'blue', 'red_blue'):
        for d in args.distances:
            for dx in args.offsets:
                for dy in args.offsets:
                    for L in args.lights:
                        flip = block_class == 'red_blue' and dx < 0 and (L == args.lights[0])
                        t = make_one(args.out, block_class, d, dx, dy, args.real_edge, args.fx, args.fy, cx, cy, args.cam_w, args.cam_h, L, args.noise, args.blur, args.background, flip, args.jpeg_quality)
                        rows.append(t)
    csv_path = os.path.join(args.out, 'ground_truth.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[OK] generated {len(rows)} images into: {os.path.abspath(args.out)}')
    print(f'[OK] ground truth csv: {os.path.abspath(csv_path)}')
    print('')
    print('next step - run the offline self test:')
    print(f'  python selftest.py --samples {args.out}')
    print('')
    print('or tune HSV thresholds interactively:')
    print(f"  python tune_hsv.py --image {os.path.join(args.out, rows[0]['file'])}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
