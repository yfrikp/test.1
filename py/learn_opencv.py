#!/usr/bin/env python3
import os
import numpy as np
import cv2
OUT = '/mnt/c/Users/Lenovo/dsh-workspace/learn_out'

def step(n, title):
    print(f'\n===== 第 {n} 步：{title} =====')
step(0, '造一张测试图（左边红块，右边蓝块）')
os.makedirs(OUT, exist_ok=True)
img = np.full((480, 640, 3), 40, np.uint8)
cv2.rectangle(img, (120, 160), (280, 320), (0, 0, 255), -1)
cv2.rectangle(img, (380, 200), (500, 320), (255, 0, 0), -1)
print(f'  图的大小 img.shape = {img.shape}   →  (高, 宽, 通道数)，不是(宽,高)')
cv2.imwrite(f'{OUT}/0_原图.png', img)
step(1, '把颜色从 BGR 转到 HSV')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print('  为什么必须转：HSV 把「颜色」和「亮度」分开了，开灯关灯只影响 V，H 不变')
print('  记住：OpenCV 的 H 是 0~179（不是 0~360），S/V 是 0~255')
step(2, '用阈值把红色「抠」出来（红色要写两段区间）')
red_low = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
red_high = cv2.inRange(hsv, (170, 120, 120), (179, 255, 255))
red = cv2.bitwise_or(red_low, red_high)
blue = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
print(f'  红像素 = {int(red.sum() / 255)}    蓝像素 = {int(blue.sum() / 255)}')
print('  inRange 之后，符合颜色的像素变成 255（白），不符合的变成 0（黑）')
cv2.imwrite(f'{OUT}/1_红色掩膜.png', red)
cv2.imwrite(f'{OUT}/2_蓝色掩膜.png', blue)
step(3, '在掩膜上找轮廓（就是物体的边界线）')
(cnts, _) = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'  找到 {len(cnts)} 个红色区域')
print('  注意：新版 OpenCV 只返回 2 个值，照抄老教程写 3 个会直接报错')
step(4, '量出最大那块的中心和边长')
c = max(cnts, key=cv2.contourArea)
((cx, cy), (w, h), angle) = cv2.minAreaRect(c)
print(f'  中心 = ({cx:.0f}, {cy:.0f})')
print(f'  边长 = {w:.0f} x {h:.0f} 像素')
print('  预期结果：中心 = (200, 240)，边长 = 160 x 160')
step(5, '把识别结果画在原图上并存盘')
vis = img.copy()
box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int)
cv2.drawContours(vis, [box], 0, (0, 255, 0), 3)
cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)
cv2.putText(vis, 'RED', (int(cx) - 40, int(cy) - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imwrite(f'{OUT}/3_识别结果.png', vis)
print(f'  已保存：{OUT}/3_识别结果.png')
step(6, '像素 → 厘米 / 距离（题目1 要 3cm 误差，靠这一步）')
real_cm = 5.0
px_per_cm = w / real_cm
print(f'  假设红块真实边长 {real_cm} cm → 1 厘米 = {px_per_cm:.1f} 像素')
f_px = 600.0
z_cm = f_px * real_cm / w
print(f'  按相似三角形 Z = f × 真实边长 ÷ 像素边长 → 距离约 {z_cm:.1f} cm')
print('\n' + '=' * 56)
print('全部完成 ✅  结果图片在：')
print('  C:\\Users\\Lenovo\\dsh-workspace\\learn_out\\   （资源管理器打开这个文件夹）')
print('=' * 56)
