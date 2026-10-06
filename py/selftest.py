from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DetectorParams, detect

def read_ground_truth(samples_dir: str) -> List[Dict[str, object]]:
    csv_path = os.path.join(samples_dir, 'ground_truth.csv')
    rows: List[Dict[str, object]] = []
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows
    for fn in sorted(os.listdir(samples_dir)):
        if not fn.endswith('.txt'):
            continue
        d: Dict[str, object] = {}
        with open(os.path.join(samples_dir, fn), 'r', encoding='utf-8') as f:
            for line in f:
                if ':' not in line:
                    continue
                (k, v) = line.split(':', 1)
                d[k.strip()] = v.strip()
        if 'file' in d:
            rows.append(d)
    return rows

def fnum(row: Dict[str, object], key: str, default: float=0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default

def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))

def main() -> int:
    ap = argparse.ArgumentParser(description='Offline self test for vision_task1 detector')
    ap.add_argument('--samples', default='samples', help='directory with synthetic images')
    ap.add_argument('--edge', type=float, default=None, help='real_edge_m to use; default = value from ground truth')
    ap.add_argument('--repeat', type=int, default=120, help='how many frames to time for the FPS estimate')
    ap.add_argument('--save-fail', default='', help='optional dir to save images that were misclassified')
    ap.add_argument('--limit', type=int, default=0, help='only test first N images (0=all)')
    args = ap.parse_args()
    if not os.path.isdir(args.samples):
        print(f'[FAIL] samples directory not found: {args.samples}')
        print('       run first:  python make_test_images.py --out samples')
        return 2
    rows = read_ground_truth(args.samples)
    if not rows:
        print(f'[FAIL] no ground truth found in {args.samples}')
        return 2
    edge_from_truth = fnum(rows[0], 'real_edge_m', 0.05)
    edge = args.edge if args.edge is not None else edge_from_truth
    p = DetectorParams()
    p.real_edge_m = edge
    (p.fx, p.fy, p.cx, p.cy) = (600.0, 600.0, 320.0, 240.0)
    if args.limit > 0:
        rows = rows[:args.limit]
    print('=' * 74)
    print('  vision_task1 offline self test')
    print('=' * 74)
    print(f'  samples dir      : {os.path.abspath(args.samples)}')
    print(f'  images           : {len(rows)}')
    print(f'  real_edge_m used : {edge}   (from ground truth: {edge_from_truth})')
    print(f'  intrinsics       : fx={p.fx} fy={p.fy} cx={p.cx} cy={p.cy}')
    print('')
    if abs(edge - edge_from_truth) > 1e-09:
        print('  [WARN] --edge differs from the value used to generate the images.')
        print('         Distance error will include this mismatch (that is realistic,')
        print('         but for verifying the formula you want them equal).')
        print('')
    total = 0
    correct = 0
    no_target = 0
    conf_mat: Dict[Tuple[str, str], int] = {}
    z_err: List[float] = []
    x_err: List[float] = []
    y_err: List[float] = []
    e_err: List[float] = []
    z_err_pct: List[float] = []
    worst: List[Tuple[float, str, str, float, float]] = []
    os.makedirs(args.save_fail, exist_ok=True) if args.save_fail else None
    for row in rows:
        fn = str(row.get('file', ''))
        img_path = os.path.join(args.samples, fn)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'  [WARN] cannot read {fn}')
            continue
        truth_cls = str(row.get('block_class', ''))
        tz = fnum(row, 'distance_m')
        tx = fnum(row, 'x_m')
        ty = fnum(row, 'y_m')
        det = detect(img, p)
        total += 1
        conf_mat[truth_cls, det.block_class] = conf_mat.get((truth_cls, det.block_class), 0) + 1
        if det.block_class == 'unknown':
            no_target += 1
        if det.block_class == truth_cls:
            correct += 1
        if args.save_fail and det.block_class != truth_cls:
            cv2.imwrite(os.path.join(args.save_fail, f'FAIL_{truth_cls}_as_{det.block_class}_{fn}'), img)
        if det.block_class == truth_cls and det.z_m > 0:
            dz = det.z_m - tz
            dxe = det.x_m - tx
            dye = det.y_m - ty
            de = det.edge_length_m - edge
            z_err.append(dz)
            x_err.append(dxe)
            y_err.append(dye)
            e_err.append(de)
            if tz > 1e-09:
                z_err_pct.append(abs(dz) / tz * 100.0)
            worst.append((abs(dz), fn, f'{truth_cls}->{det.block_class}', tz, det.z_m))
    if total == 0:
        print('[FAIL] no images were processed')
        return 2
    sample_img = None
    for row in rows:
        im = cv2.imread(os.path.join(args.samples, str(row.get('file', ''))), cv2.IMREAD_COLOR)
        if im is not None:
            sample_img = im
            break
    fps = 0.0
    ms_per_frame = 0.0
    if sample_img is not None and args.repeat > 0:
        detect(sample_img, p)
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            detect(sample_img, p)
        dt = time.perf_counter() - t0
        ms_per_frame = dt / args.repeat * 1000.0
        fps = args.repeat / dt if dt > 0 else 0.0
    acc = correct / total * 100.0
    print('-' * 74)
    print('  1) CLASSIFICATION')
    print('-' * 74)
    print(f'  accuracy          : {correct}/{total} = {acc:.1f}%')
    if no_target:
        print(f'  missed (unknown)  : {no_target}  <-- threshold too strict / block too small')
    print('')
    print('  confusion matrix (truth -> predicted):')
    classes = ['red', 'blue', 'red_blue', 'unknown']
    print('    {:<12}'.format('truth\\pred') + ''.join(('{:>11}'.format(c) for c in classes)))
    for t in classes:
        row_counts = [conf_mat.get((t, c), 0) for c in classes]
        if sum(row_counts) == 0:
            continue
        print('    {:<12}'.format(t) + ''.join(('{:>11}'.format(v) for v in row_counts)))
    print('')
    print('-' * 74)
    print('  2) ACCURACY  (requirement: |error| <= 3 cm)')
    print('-' * 74)
    if z_err:
        az = [abs(v) for v in z_err]
        ax = [abs(v) for v in x_err]
        ay = [abs(v) for v in y_err]
        print(f'  samples with valid pose : {len(z_err)}')
        print('')
        print('    {:<22}{:>12}{:>12}{:>12}{:>12}'.format('quantity', 'mean|err|', 'p95|err|', 'max|err|', 'mean signed'))
        print('    ' + '-' * 70)
        print('    {:<22}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}'.format('distance Z (m)', float(np.mean(az)), pct(az, 95), float(np.max(az)), float(np.mean(z_err))))
        print('    {:<22}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}'.format('offset X (m)', float(np.mean(ax)), pct(ax, 95), float(np.max(ax)), float(np.mean(x_err))))
        print('    {:<22}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}'.format('offset Y (m)', float(np.mean(ay)), pct(ay, 95), float(np.max(ay)), float(np.mean(y_err))))
        print('    {:<22}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}'.format('edge length (m)', float(np.mean([abs(v) for v in e_err])), pct([abs(v) for v in e_err], 95), float(np.max([abs(v) for v in e_err])), float(np.mean(e_err))))
        print('')
        print(f'    mean |Z| relative error : {float(np.mean(z_err_pct)):.3f} %')
        print(f'    max  |Z| error          : {float(np.max(az)) * 100:.3f} cm')
        verdict = 'PASS' if float(np.max(az)) <= 0.03 else 'FAIL'
        print(f'    [3cm requirement]       : {verdict}')
        print('')
        worst.sort(reverse=True)
        print('  worst 5 distance errors (|err| m, file, class, truth Z, measured Z):')
        for w in worst[:5]:
            print(f'    {w[0]:.6f}  {w[1]:<46} {w[2]:<16} {w[3]:.4f} -> {w[4]:.4f}')
    else:
        print('  no valid pose samples (all detections were unknown or misclassified)')
    print('')
    print('-' * 74)
    print('  3) SPEED')
    print('-' * 74)
    if fps > 0:
        print(f'  per-frame (CPU, single thread) : {ms_per_frame:.3f} ms')
        print(f'  theoretical rate               : {fps:.1f} Hz')
        needed = 1000.0 / 60.0
        print(f'  budget for 60 Hz               : {needed:.3f} ms/frame')
        verdict = 'PASS' if ms_per_frame <= needed else 'FAIL'
        print(f'  [60Hz requirement]             : {verdict}')
        if ms_per_frame > needed:
            print('')
            print('  note: python + opencv on CPU is usually slower than C++.')
            print('        the assessment only needs the ROS2 topic at >=60Hz, and the')
            print('        published rate can be decoupled from the camera/detect rate.')
            print('        the C++ node is the one you submit for the 60Hz requirement.')
    print('')
    print('=' * 74)
    overall = acc >= 99.0 and (not z_err or float(np.max([abs(v) for v in z_err])) <= 0.03)
    print(f"  OVERALL: {('PASS' if overall else 'NEEDS WORK')}   (classification {acc:.1f}%)")
    print('=' * 74)
    return 0
if __name__ == '__main__':
    sys.exit(main())
