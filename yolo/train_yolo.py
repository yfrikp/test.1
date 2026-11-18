#!/usr/bin/env python3
"""题目1 加分项②：YOLO 调优。

用 samples_synth 的 324 张带真值合成图自动生成 YOLO 数据集，
微调 yolo11n，输出训练指标、与 OpenCV 方案的对比，并留下可复现脚本。

运行：python3 yolo/train_yolo.py
"""
import csv
import json
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(ROOT, 'samples_synth')
GT = os.path.join(SYNTH, 'ground_truth.csv')
OUT = os.path.join(ROOT, 'yolo')
DS = os.path.join(OUT, 'dataset')
CLASSES = ['red', 'blue', 'red_blue']
IMG_W, IMG_H = 640, 480


def build_dataset():
    if os.path.isdir(DS):
        shutil.rmtree(DS)
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(DS, sub), exist_ok=True)

    random.seed(0)
    rows = []
    with open(GT, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: r['file'])

    n_train = 0
    for i, r in enumerate(rows):
        src = os.path.join(SYNTH, r['file'])
        if not os.path.exists(src):
            continue
        split = 'val' if i % 5 == 0 else 'train'   # 258 / 66
        shutil.copy2(src, os.path.join(DS, 'images', split, r['file']))
        cls = CLASSES.index(r['block_class'])
        u, v = float(r['u_px']), float(r['v_px'])
        w = h = float(r['edge_px_drawn'])
        label = (f'{cls} {u / IMG_W:.6f} {v / IMG_H:.6f} '
                 f'{w / IMG_W:.6f} {h / IMG_H:.6f}\n')
        name = os.path.splitext(r['file'])[0] + '.txt'
        with open(os.path.join(DS, 'labels', split, name), 'w', encoding='utf-8') as f:
            f.write(label)
        if split == 'train':
            n_train += 1
    yaml_path = os.path.join(DS, 'blocks.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {DS}\ntrain: images/train\nval: images/val\n'
                f'nc: {len(CLASSES)}\nnames: {CLASSES}\n')
    print(f'[dataset] 训练 {n_train} 张，验证 {len(rows) - n_train} 张 -> {yaml_path}')
    return yaml_path


def main():
    yaml_path = build_dataset()
    from ultralytics import YOLO

    base = os.path.join(ROOT, 'models', 'yolo11n.pt')
    if not os.path.exists(base):
        base = 'yolo11n.pt'
    model = YOLO(base)
    t0 = time.time()
    model.train(
        data=yaml_path, epochs=60, imgsz=480, batch=16, device=0,
        project=os.path.join(OUT, 'runs'), name='blocks', exist_ok=True,
        patience=20, workers=4, seed=0, verbose=False,
        degrees=0.0, fliplr=0.0, flipud=0.0, mosaic=0.0,
        plots=False, val=True, save=True,
    )
    train_s = time.time() - t0

    best = os.path.join(OUT, 'runs', 'blocks', 'weights', 'best.pt')
    print(f'[train] 完成，用时 {train_s:.0f}s，权重 {best}')

    m = YOLO(best)
    metrics = m.val(data=yaml_path, imgsz=480, device=0, verbose=False, plots=False)
    report = {
        'epochs': 60,
        'imgsz': 480,
        'train_images': 258,
        'val_images': 66,
        'classes': CLASSES,
        'train_seconds': round(train_s, 1),
        'mAP50': round(float(metrics.box.map50), 4),
        'mAP50_95': round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall': round(float(metrics.box.mr), 4),
    }

    # 推理耗时 / 帧率
    import glob
    imgs = sorted(glob.glob(os.path.join(DS, 'images', 'val', '*.png')))[:40]
    for _ in range(5):
        m.predict(imgs[0], imgsz=480, device=0, verbose=False)
    t0 = time.time()
    for p in imgs:
        m.predict(p, imgsz=480, device=0, conf=0.25, verbose=False)
    ms = (time.time() - t0) / len(imgs) * 1000
    report['infer_ms_per_image'] = round(ms, 2)
    report['infer_fps'] = round(1000.0 / ms, 1)

    with open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('[metrics] ' + json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
