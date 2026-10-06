#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 第一课：从一张图里找出红块，量出中心点和边长
================================================================
用法（在「① Ubuntu 终端」里跑，不是在 ⑥ 的 Python 窗口里）：

    python3 /mnt/c/Users/Lenovo/dsh-workspace/vision_task1/py/learn_opencv.py

它会把每一步的中间结果都存成图片，放在
    C:\\Users\\Lenovo\\dsh-workspace\\learn_out\\
这样就算弹窗有问题，你也能在 Windows 资源管理器里直接打开图片看结果。
================================================================
"""
import os
import numpy as np
import cv2

OUT = '/mnt/c/Users/Lenovo/dsh-workspace/learn_out'


def step(n, title):
    print(f'\n===== 第 {n} 步：{title} =====')


# ---------------------------------------------------------------- 第 0 步
step(0, '造一张测试图（左边红块，右边蓝块）')
os.makedirs(OUT, exist_ok=True)
img = np.full((480, 640, 3), 40, np.uint8)                    # 深灰底：480 高、640 宽、3 通道
cv2.rectangle(img, (120, 160), (280, 320), (0, 0, 255), -1)   # 红块 —— 注意 BGR，红色是 (0,0,255)
cv2.rectangle(img, (380, 200), (500, 320), (255, 0, 0), -1)   # 蓝块
print(f'  图的大小 img.shape = {img.shape}   →  (高, 宽, 通道数)，不是(宽,高)')
cv2.imwrite(f'{OUT}/0_原图.png', img)

# ---------------------------------------------------------------- 第 1 步
step(1, '把颜色从 BGR 转到 HSV')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print('  为什么必须转：HSV 把「颜色」和「亮度」分开了，开灯关灯只影响 V，H 不变')
print('  记住：OpenCV 的 H 是 0~179（不是 0~360），S/V 是 0~255')

# ---------------------------------------------------------------- 第 2 步
step(2, '用阈值把红色「抠」出来（红色要写两段区间）')
red_low = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))     # 红色在 H=0 附近
red_high = cv2.inRange(hsv, (170, 120, 120), (179, 255, 255))  # 红色还跨到 H=180 附近
red = cv2.bitwise_or(red_low, red_high)                       # 两段合并
blue = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
print(f'  红像素 = {int(red.sum() / 255)}    蓝像素 = {int(blue.sum() / 255)}')
print('  inRange 之后，符合颜色的像素变成 255（白），不符合的变成 0（黑）')
cv2.imwrite(f'{OUT}/1_红色掩膜.png', red)
cv2.imwrite(f'{OUT}/2_蓝色掩膜.png', blue)

# ---------------------------------------------------------------- 第 3 步
step(3, '在掩膜上找轮廓（就是物体的边界线）')
cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'  找到 {len(cnts)} 个红色区域')
print('  注意：新版 OpenCV 只返回 2 个值，照抄老教程写 3 个会直接报错')

# ---------------------------------------------------------------- 第 4 步
step(4, '量出最大那块的中心和边长')
c = max(cnts, key=cv2.contourArea)                # 面积最大的那个轮廓
(cx, cy), (w, h), angle = cv2.minAreaRect(c)      # 最小外接旋转矩形
print(f'  中心 = ({cx:.0f}, {cy:.0f})')
print(f'  边长 = {w:.0f} x {h:.0f} 像素')
print('  预期结果：中心 = (200, 240)，边长 = 160 x 160')

# ---------------------------------------------------------------- 第 5 步
step(5, '把识别结果画在原图上并存盘')
vis = img.copy()
box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int)
cv2.drawContours(vis, [box], 0, (0, 255, 0), 3)                        # 绿色框
cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)              # 黄色中心点
cv2.putText(vis, 'RED', (int(cx) - 40, int(cy) - 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imwrite(f'{OUT}/3_识别结果.png', vis)
print(f'  已保存：{OUT}/3_识别结果.png')

# ---------------------------------------------------------------- 第 6 步
step(6, '像素 → 厘米 / 距离（题目1 要 3cm 误差，靠这一步）')
real_cm = 5.0                                   # ← 你实测的物块真实边长（厘米）
px_per_cm = w / real_cm
print(f'  假设红块真实边长 {real_cm} cm → 1 厘米 = {px_per_cm:.1f} 像素')
f_px = 600.0                                    # 相机焦距（像素），要用「已知距离」的图标定出来
z_cm = f_px * real_cm / w
print(f'  按相似三角形 Z = f × 真实边长 ÷ 像素边长 → 距离约 {z_cm:.1f} cm')

print('\n' + '=' * 56)
print('全部完成 ✅  结果图片在：')
print(r'  C:\Users\Lenovo\dsh-workspace\learn_out\   （资源管理器打开这个文件夹）')
print('=' * 56)
