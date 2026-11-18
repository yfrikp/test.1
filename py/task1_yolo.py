#!/usr/bin/env python3
"""题目1 加分项②：YOLO 分支。

用 yolo/train_yolo.py 微调出来的模型做识别，并把 bbox 换算成和 OpenCV 分支
一样的输出（种类 / 边长 / 距光心坐标），可以单独跑，也可以拿来和 OpenCV 分支对比。

用法：
  python3 py/task1_yolo.py --src samples/first.png --show          # 单张图
  python3 py/task1_yolo.py --eval                                  # 在 65 张带真值的验证集上算误差
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, 'yolo', 'runs', 'blocks', 'weights', 'best.pt')
CLASSES = ['red', 'blue', 'red_blue']


def load_model(weights):
    from ultralytics import YOLO
    if not os.path.exists(weights):
        sys.exit(f'找不到权重 {weights}，先跑 python3 yolo/train_yolo.py')
    return YOLO(weights)


def pick_box(result, conf):
    """取置信度最高、且类别在白名单里的框。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    for i in range(len(boxes)):
        c = float(boxes.conf[i])
        if c < conf:
            continue
        if best is None or c > best[0]:
            best = (c, i)
    return None if best is None else best[1]


def box_to_info(box_xyxy, cls_id, conf, fx, fy, cx, cy, real_edge):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w, h = x2 - x1, y2 - y1
    edge_px = (w + h) / 2.0
    z = fx * real_edge / max(edge_px, 1e-6)
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return dict(block_class=CLASSES[int(cls_id)], edge_length_m=real_edge,
                edge_length_px=edge_px, x_m=x, y_m=y, z_m=z, u_px=u, v_px=v,
                confidence=conf)


def run_single(model, src, conf, fx, fy, cx, cy, real_edge, show, out_dir):
    import cv2
    paths = sorted(glob.glob(src)) if any(ch in src for ch in '*?[') else [src]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f'[跳过] 读不到 {p}')
            continue
        h, w = img.shape[:2]
        cx_, cy_ = (cx if cx else w / 2.0), (cy if cy else h / 2.0)
        res = model.predict(p, imgsz=480, conf=conf, verbose=False)[0]
        idx = pick_box(res, conf)
        print(f'\n=== {os.path.basename(p)}  {w}x{h} ===')
        if idx is None:
            print('  未检出目标')
            continue
        info = box_to_info(res.boxes.xyxy[idx], float(res.boxes.cls[idx]),
                           float(res.boxes.conf[idx]), fx, fy, cx_, cy_, real_edge)
        print(f"  种类 {info['block_class']}  置信度 {info['confidence']:.3f}")
        print(f"  边长 {info['edge_length_px']:.1f} px -> {info['edge_length_m'] * 100:.1f} cm")
        print(f"  距光心 x={info['x_m'] * 100:+.2f} cm  y={info['y_m'] * 100:+.2f} cm  z={info['z_m'] * 100:.2f} cm")
        x1, y1, x2, y2 = [int(v) for v in res.boxes.xyxy[idx]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{info['block_class']} {info['z_m']:.2f}m",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, 'yolo_' + os.path.basename(p))
            cv2.imwrite(dst, img)
            print(f'  已保存 {dst}')
        if show:
            cv2.imshow('yolo', img)
            cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()


def run_eval(model, conf, fx, fy, cx, cy, real_edge):
    syn = os.path.join(ROOT, 'samples_synth')
    gt_path = os.path.join(syn, 'ground_truth.csv')
    rows = [r for r in csv.DictReader(open(gt_path, newline='', encoding='utf-8'))]
    val = [r for i, r in enumerate(sorted(rows, key=lambda r: r['file'])) if i % 5 == 0]
    ok_cls = ok_3cm = 0
    errs = []
    miss = 0
    for r in val:
        p = os.path.join(syn, r['file'])
        img = __import__('cv2').imread(p)
        h, w = img.shape[:2]
        cx_, cy_ = (cx if cx else w / 2.0), (cy if cy else h / 2.0)
        res = model.predict(p, imgsz=480, conf=conf, verbose=False)[0]
        idx = pick_box(res, conf)
        if idx is None:
            miss += 1
            continue
        info = box_to_info(res.boxes.xyxy[idx], float(res.boxes.cls[idx]),
                           float(res.boxes.conf[idx]), fx, fy, cx_, cy_, real_edge)
        if info['block_class'] == r['block_class']:
            ok_cls += 1
        err = abs(info['z_m'] - float(r['distance_eff_m'])) * 100.0
        errs.append(err)
        if err <= 3.0:
            ok_3cm += 1
    n = len(val)
    print(f'\n验证集 {n} 张（与训练时同一划分：每 5 张取 1）')
    print(f'  类别正确 {ok_cls}/{n} = {ok_cls / n * 100:.1f}%')
    print(f'  未检出 {miss} 张')
    print(f'  距离误差 <=3cm：{ok_3cm}/{n} = {ok_3cm / n * 100:.1f}%')
    if errs:
        e = np.array(errs)
        print(f'  距离误差 平均 {e.mean():.2f} cm  p95 {np.percentile(e, 95):.2f} cm  最大 {e.max():.2f} cm')


def main():
    ap = argparse.ArgumentParser(description='题目1 加分项②：YOLO 识别与误差评测')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--src', default='samples/first.png')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--fx', type=float, default=600.0)
    ap.add_argument('--fy', type=float, default=600.0)
    ap.add_argument('--cx', type=float, default=0.0, help='0 = 用图像宽度的一半')
    ap.add_argument('--cy', type=float, default=0.0, help='0 = 用图像高度的一半')
    ap.add_argument('--real-edge', type=float, default=0.05)
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--eval', action='store_true', help='在带真值的验证集上算误差')
    args = ap.parse_args()

    model = load_model(args.weights)
    if args.eval:
        run_eval(model, args.conf, args.fx, args.fy, args.cx, args.cy, args.real_edge)
    else:
        run_single(model, args.src, args.conf, args.fx, args.fy, args.cx, args.cy,
                   args.real_edge, args.show, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
