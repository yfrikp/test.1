#!/usr/bin/env python3
import argparse
import os
import sys
import time
import cv2
import numpy as np
P = dict(red_h_low_max=8, red_h_high_min=172, red_s_min=90, red_v_min=60, blue_h_min=100, blue_h_max=132, blue_s_min=70, blue_v_min=60, open_kernel=3, close_kernel=13, min_area_px=200, min_area_ratio=0.0005, dominant_ratio=0.8, interleave_ratio=0.18, layer_purity=0.85, layer_tol=0.25, real_edge_m=0.05, fx=600.0, fy=600.0, cx=320.0, cy=240.0, dist=[0.0, 0.0, 0.0, 0.0, 0.0], offset_x_m=0.0, offset_y_m=0.0, measure='area', edge_bias_px=1.0)

def measure_edge_px(cnt, p):
    (x, y, w, h) = cv2.boundingRect(cnt)
    bbox_est = (w + h) / 2.0
    ((_, _), (rw, rh), _) = cv2.minAreaRect(cnt)
    minrect_est = max(rw, rh)
    area_est = float(np.sqrt(max(0.0, cv2.contourArea(cnt)))) - float(p['edge_bias_px'])
    face_est = None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(float)
        sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
        if min(sides) > 0 and max(sides) / min(sides) < 1.35:
            face_est = sum(sides) / 4.0
    mode = p['measure']
    common = (bbox_est, minrect_est, face_est, area_est)
    if mode == 'bbox':
        return (bbox_est, 'bbox', *common)
    if mode == 'minrect':
        return (minrect_est, 'minrect', *common)
    if mode == 'front_face':
        if face_est:
            return (face_est, 'front_face', *common)
        return (area_est, 'area(退)', *common)
    if mode == 'auto':
        if face_est:
            return (face_est, 'front_face', *common)
        return (area_est, 'area', *common)
    return (area_est, 'area', *common)

def _best_split_1d(cnt_a, cnt_b, total, min_layer):
    best = None
    n = len(cnt_a)
    for i in range(1, n):
        (a1, b1) = (sum(cnt_a[:i]), sum(cnt_b[:i]))
        (a2, b2) = (sum(cnt_a[i:]), sum(cnt_b[i:]))
        for (cost, first_is_red, m1, m2) in ((b1 + a2, True, a1, b2), (a1 + b2, False, b1, a2)):
            if m1 < min_layer * total or m2 < min_layer * total:
                continue
            if best is None or cost < best[0]:
                best = (cost, first_is_red, m1, m2)
    return best

def _layer_split(blob_red, blob_blue, box, min_layer=0.12):
    (x, y, w, h) = box
    rows_r = [int(np.count_nonzero(blob_red[r, x:x + w])) for r in range(y, y + h)]
    rows_b = [int(np.count_nonzero(blob_blue[r, x:x + w])) for r in range(y, y + h)]
    total = sum(rows_r) + sum(rows_b)
    if total == 0:
        return (None, None, 0.0, 0.0, 0, 0, 0)
    cols_r = [int(np.count_nonzero(blob_red[y:y + h, c])) for c in range(x, x + w)]
    cols_b = [int(np.count_nonzero(blob_blue[y:y + h, c])) for c in range(x, x + w)]
    cands = []
    bh = _best_split_1d(rows_r, rows_b, total, min_layer)
    if bh:
        cands.append((bh[0], 'h', bh[1], bh[2], bh[3]))
    bv = _best_split_1d(cols_r, cols_b, total, min_layer)
    if bv:
        cands.append((bv[0], 'v', bv[1], bv[2], bv[3]))
    if not cands:
        return (None, None, 0.0, 0.0, 0, 0, total)
    (cost, axis, first_is_red, m1, m2) = min(cands, key=lambda t: t[0])
    purity = 1.0 - cost / total
    balance = min(m1, m2) / max(m1, m2) if max(m1, m2) > 0 else 0.0
    return (axis, first_is_red, purity, balance, m1, m2, total)

def classify(blob_red, blob_blue, box, p):
    n_red = int(np.count_nonzero(blob_red))
    n_blue = int(np.count_nonzero(blob_blue))
    total = n_red + n_blue
    if total == 0:
        return ('unknown', 0.0)
    r_red = n_red / total
    r_blue = n_blue / total
    if r_red >= p['dominant_ratio']:
        return ('red', r_red)
    if r_blue >= p['dominant_ratio']:
        return ('blue', r_blue)
    if r_red >= p['interleave_ratio'] and r_blue >= p['interleave_ratio']:
        (axis, first_is_red, purity, balance, m1, m2, _) = _layer_split(blob_red, blob_blue, box)
        if axis is not None and purity >= p['layer_purity'] and (balance >= p['layer_tol']):
            return ('red_blue', purity)
    return ('red' if r_red > r_blue else 'blue', max(r_red, r_blue))

def solve_pose(edge_px, box, p):
    (x, y, w, h) = box
    u = x + w / 2.0
    v = y + h / 2.0
    if edge_px <= 1.0:
        return (0.0, 0.0, 0.0, u, v)
    z = p['fy'] * p['real_edge_m'] / edge_px
    x_m = (u - p['cx']) * z / p['fx']
    y_m = (v - p['cy']) * z / p['fy']
    x_m -= p['offset_x_m']
    y_m -= p['offset_y_m']
    return (x_m, y_m, z, u, v)

def detect(frame, p):
    (H, W) = frame.shape[:2]
    if any((abs(v) > 1e-09 for v in p['dist'])):
        K = np.array([[p['fx'], 0.0, p['cx']], [0.0, p['fy'], p['cy']], [0.0, 0.0, 1.0]], dtype=np.float64)
        frame = cv2.undistort(frame, K, np.array(p['dist'], dtype=np.float64))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, p['red_s_min'], p['red_v_min']), (p['red_h_low_max'], 255, 255)) | cv2.inRange(hsv, (p['red_h_high_min'], p['red_s_min'], p['red_v_min']), (179, 255, 255))
    mask_blue = cv2.inRange(hsv, (p['blue_h_min'], p['blue_s_min'], p['blue_v_min']), (p['blue_h_max'], 255, 255))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (p['open_kernel'],) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (p['close_kernel'],) * 2)
    red_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k_open)
    blue_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k_open)
    both = cv2.bitwise_or(red_clean, blue_clean)
    both = cv2.morphologyEx(both, cv2.MORPH_CLOSE, k_close)
    (contours, _) = cv2.findContours(both, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(float(p['min_area_px']), float(p['min_area_ratio']) * H * W)
    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        box = cv2.boundingRect(cnt)
        only = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(only, [cnt], -1, 255, cv2.FILLED)
        blob_red = cv2.bitwise_and(red_clean, only)
        blob_blue = cv2.bitwise_and(blue_clean, only)
        (kind, conf) = classify(blob_red, blob_blue, box, p)
        n_red = int(np.count_nonzero(blob_red))
        n_blue = int(np.count_nonzero(blob_blue))
        (edge_px, how, bbox_est, minrect_est, face_est, area_est) = measure_edge_px(cnt, p)
        (x_m, y_m, z_m, u, v) = solve_pose(edge_px, box, p)
        results.append(dict(kind=kind, conf=conf, box=box, edge_px=edge_px, how=how, x=x_m, y=y_m, z=z_m, u=u, v=v, bbox_est=bbox_est, minrect_est=minrect_est, face_est=face_est, area_est=area_est, n_red=n_red, n_blue=n_blue, area=cv2.contourArea(cnt), cnt=cnt))
    results.sort(key=lambda r: -r['area'])
    return (results, red_clean, blue_clean)

def draw(frame, results):
    vis = frame.copy()
    for r in results:
        (x, y, w, h) = r['box']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis, r['kind'], (x, max(11, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return vis

def print_results(results, p):
    print('-' * 96)
    print(f"{'种类':<10}{'边长(cm)':>9}{'X(cm)':>9}{'Y(cm)':>9}{'距离Z(cm)':>11}{'像素边':>9}{'置信':>7}   测量方式")
    print('-' * 96)
    for r in results:
        print(f"{r['kind']:<10}{p['real_edge_m'] * 100:>9.2f}{r['x'] * 100:>9.2f}{r['y'] * 100:>9.2f}{r['z'] * 100:>11.2f}{r['edge_px']:>9.1f}{r['conf']:>7.1%}   {r['how']}")
    print('-' * 96)
    print(f'识别到 {len(results)} 个物块')
    print()
    print("【四种像素边长估计值对比】看它们差多少，就知道'边长取法'对精度影响多大：")
    print(f"  {'':4}{'种类':<11}{'bbox':>8}{'minAreaRect':>13}{'正方形面':>11}{'√面积-偏置':>12}{'→ 采用':>10}")
    for (i, r) in enumerate(results):
        face = f"{r['face_est']:.1f}" if r['face_est'] else '无(非四边形)'
        print(f"  第{i + 1}个 {r['kind']:<11}{r['bbox_est']:>8.1f}{r['minrect_est']:>13.1f}{face:>11}{r['area_est']:>12.2f}{r['edge_px']:>10.2f}   ({r['how']})")
    print('  说明：四种估计都系统性偏大（掩膜外扩），所以面积法要减 edge_bias_px。')
    print('        bbox/minAreaRect 会被轮廓噪声撑大，小目标上尤其明显；面积法最稳。')
    print()

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description='题目1 颜色物块识别与定位')
    ap.add_argument('--src', default=os.path.join(here, '..', 'samples', 'first.png'), help='图片或视频文件路径')
    ap.add_argument('--camera', type=int, default=None, help='用摄像头，传设备号（一般是 0）')
    ap.add_argument('--out', default=None, help='结果图保存路径（不填就不存）')
    ap.add_argument('--show', action='store_true', help='弹窗显示（按 q 退出）')
    ap.add_argument('--benchmark', type=int, default=0, help='重复处理 N 帧并报告帧率（用来量 60Hz 达不达标）')
    args = ap.parse_args()
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print('[错误] 打不开摄像头。注意：WSL 里通常没有 /dev/video0，')
            print('       要么用 Windows 侧采集，要么用 usbipd 直通，要么先处理图片/视频。')
            return 1
        src_desc = f'摄像头 {args.camera}'
        is_file = False
    else:
        src = os.path.abspath(args.src)
        if not os.path.exists(src):
            print(f'[错误] 找不到文件：{src}')
            return 1
        ext = os.path.splitext(src)[1].lower()
        is_file = ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        if is_file:
            img = cv2.imread(src)
            if img is None:
                print('[错误] 图片读不出来（路径有中文/空格时注意引号）')
                return 1
            src_desc = src
        else:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                print(f'[错误] 打不开视频：{src}')
                return 1
            src_desc = src
    print()
    print('=' * 96)
    print(f'输入：{src_desc}')
    print('=' * 96)
    if not args.camera and is_file:
        (H, W) = img.shape[:2]
        if abs(P['cx'] - W / 2) > 0.2 * W or abs(P['cy'] - H / 2) > 0.2 * H:
            print(f"[提醒] 这张图是 {W}x{H}，但参数里 cx={P['cx']:.0f} cy={P['cy']:.0f} 是按 640x480 填的。")
            print('       → 算出来的 X/Y 会明显偏移，请换成这张图真正的内参（标定得到）。')
            print()
        (results, _, _) = detect(img, P)
        print_results(results, P)
        if args.out:
            out_path = os.path.abspath(args.out)
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            ok = cv2.imwrite(out_path, draw(img, results))
            print(f"结果图已保存：{out_path}  ({('成功' if ok else '失败')})")
        if args.benchmark > 0:
            t0 = time.time()
            for _ in range(args.benchmark):
                detect(img, P)
            dt = time.time() - t0
            fps = args.benchmark / dt
            print(f'\n【帧率测试】处理 {args.benchmark} 帧用 {dt:.3f}s  →  {fps:.1f} FPS')
            print(f"  {('达标' if fps >= 60 else '不达标')}（题目要求 >= 60Hz）")
            if fps < 60:
                print('  提不上去的常见原因：图片分辨率太大（先降到 640x480）、')
                print('  没开 Release 优化（Python 影响小，C++ 节点影响很大）、')
                print('  同一帧里物块太多、形态学核开得太大。')
        if args.show:
            cv2.imshow('result', draw(img, results))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0
    n_frame = 0
    t_start = time.time()
    last_results = []
    try:
        while True:
            (ok, frame) = cap.read()
            if not ok:
                break
            n_frame += 1
            (last_results, _, _) = detect(frame, P)
            if n_frame % 30 == 0:
                el = time.time() - t_start
                print(f'  已处理 {n_frame} 帧，平均 {n_frame / el:.1f} FPS，当前画面 {len(last_results)} 个物块')
            if args.show:
                cv2.imshow('result', draw(frame, last_results))
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
    el = time.time() - t_start
    print()
    print('=' * 96)
    if n_frame > 0:
        print(f'处理了 {n_frame} 帧，总耗时 {el:.2f}s  →  {n_frame / el:.1f} FPS')
        print(f"  {('达标' if n_frame / el >= 60 else '不达标')}（题目要求 >= 60Hz）")
    print_results(last_results, P)
    return 0
if __name__ == '__main__':
    sys.exit(main())
