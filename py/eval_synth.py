#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用「带完整真值的合成图集」评测 task1_detect.py 的精度
=============================================================================
这就是"有参数"的价值：因为每张图的真值（类别、距离、偏移、真实边长）都已知，
所以能算出真实误差，而不是靠感觉说"应该挺准"。

用法：
    # 1) 先生成图集（会输出 ground_truth.csv）
    python3 make_test_images.py --out ../samples_synth

    # 2) 评测
    python3 eval_synth.py --samples ../samples_synth
=============================================================================
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task1_detect as td            # noqa: E402

CLASSES = ("red", "blue", "red_blue")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="合成图集精度评测")
    ap.add_argument("--samples", default=os.path.join(here, "..", "samples_synth"))
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 张（调试用）")
    args = ap.parse_args()

    csv_path = os.path.join(args.samples, "ground_truth.csv")
    if not os.path.exists(csv_path):
        print(f"找不到 {csv_path}，请先运行 make_test_images.py")
        return 1
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # 合成图的内参和生成时保持一致（make_test_images.py 的默认值）
    td.P["fx"], td.P["fy"] = 600.0, 600.0
    td.P["cx"], td.P["cy"] = 320.0, 240.0
    td.P["real_edge_m"] = float(rows[0]["real_edge_m"])

    print("=" * 78)
    print(f"  评测样本：{len(rows)} 张      真实边长：{td.P['real_edge_m'] * 100:.1f} cm")
    print(f"  内参：fx={td.P['fx']:.0f} fy={td.P['fy']:.0f} cx={td.P['cx']:.0f} cy={td.P['cy']:.0f}")
    print("=" * 78)

    conf = {t: {p: 0 for p in CLASSES + ("unknown",)} for t in CLASSES}
    dz, dx, dy = [], [], []
    per_dist = {}
    n_none = 0

    for r in rows:
        img = cv2.imread(os.path.join(args.samples, r["file"]))
        if img is None:
            continue
        res, _, _ = td.detect(img, td.P)

        truth_cls = r["block_class"]
        # ★ 真值必须用 distance_eff_m（按"实际画出的整数像素"反算的等效距离），
        #   不能用 distance_m（理想值）。见 make_test_images.py 里 draw_block 的说明：
        #   像素只能取整，ideal 37.5px 实际画成 38px，该图物理上对应的就是 0.789m。
        #   用 ideal 当真值会让被测算法背黑锅（实测虚报 1.59cm 的系统误差）。
        d_nom = float(r["distance_m"])
        d_true = float(r["distance_eff_m"]) if r.get("distance_eff_m") else d_nom
        per_dist.setdefault(d_nom, [])

        if not res:
            conf[truth_cls]["unknown"] += 1
            n_none += 1
            continue

        best = res[0]                      # 一张合成图里只有一个物块
        conf[truth_cls][best["kind"] if best["kind"] in CLASSES else "unknown"] += 1
        if best["z"] <= 0:
            n_none += 1
            continue

        err_z = best["z"] - d_true
        dz.append(err_z)
        dx.append(best["x"] - float(r["x_m"]))
        dy.append(best["y"] - float(r["y_m"]))
        per_dist[d_nom].append(abs(err_z))

    # ---------------- 1) 分类 ----------------
    total = sum(sum(v.values()) for v in conf.values())
    correct = sum(conf[t][t] for t in CLASSES)
    print("\n【1】分类准确率")
    print("-" * 78)
    print(f"  正确 {correct}/{total} = {correct / total:.1%}")
    header = "真值\\预测"          # Python 3.10 的 f-string 里不能出现反斜杠，所以先存成变量
    print(f"  {header:<12}{'red':>8}{'blue':>8}{'red_blue':>10}{'unknown':>9}")
    for t in CLASSES:
        print(f"  {t:<12}{conf[t]['red']:>8}{conf[t]['blue']:>8}"
              f"{conf[t]['red_blue']:>10}{conf[t]['unknown']:>9}")

    # ---------------- 2) 精度 ----------------
    def rep(name, arr):
        if not arr:
            print(f"  {name:<16} 无有效样本")
            return
        a = np.abs(np.array(arr))
        print(f"  {name:<16} 平均|误差|={a.mean() * 100:7.2f}cm   "
              f"p95={np.percentile(a, 95) * 100:7.2f}cm   最大={a.max() * 100:7.2f}cm")

    print("\n【2】定位精度（题目要求 |误差| <= 3cm）")
    print("-" * 78)
    rep("距离 Z", dz)
    rep("横向 X", dx)
    rep("纵向 Y", dy)
    if dz:
        a = np.abs(np.array(dz))
        ok = int((a <= 0.03).sum())
        print(f"\n  <=3cm 的比例：{ok}/{len(dz)} = {ok / len(dz):.1%}")

    # ---------------- 3) 按距离分组（看误差怎么随距离变化）----------------
    print("\n【3】误差随距离的变化（★这就是'误差随距离平方放大'的实证）")
    print("-" * 78)
    print("  注：真值用的是 distance_eff_m（按实际画出的像素反算），不是标称距离。")
    print(f"  {'标称距离':>10}{'样本数':>8}{'平均|Z误差|':>14}{'最大|Z误差|':>14}{'是否达标':>10}")
    for d in sorted(per_dist):
        e = np.abs(np.array(per_dist[d])) if per_dist[d] else np.array([0.0])
        ok = "达标" if e.max() <= 0.03 else "超标"
        print(f"  {d * 100:>8.0f}cm{len(per_dist[d]):>8}{e.mean() * 100:>12.2f}cm"
              f"{e.max() * 100:>12.2f}cm{ok:>10}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
