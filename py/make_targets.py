#!/usr/bin/env python3
import os
import cv2
import numpy as np
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'targets'))

def save(img, name, desc):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f'  {name:<28} {img.shape[1]}x{img.shape[0]}   {desc}')

def single_block(kind, size=1600, block=1200):
    img = np.full((size, size, 3), 255, np.uint8)
    x0 = (size - block) // 2
    y0 = (size - block) // 2
    (x1, y1) = (x0 + block, y0 + block)
    if kind == 'red':
        img[y0:y1, x0:x1] = RED
    elif kind == 'blue':
        img[y0:y1, x0:x1] = BLUE
    elif kind == 'red_blue_h':
        mid = y0 + block // 2
        img[y0:mid, x0:x1] = RED
        img[mid:y1, x0:x1] = BLUE
    elif kind == 'red_blue_v':
        mid = x0 + block // 2
        img[y0:y1, x0:mid] = RED
        img[y0:y1, mid:x1] = BLUE
    return img

def all_three(width=2400, height=900, block=520, gap=140):
    img = np.full((height, width, 3), 255, np.uint8)
    total = 3 * block + 2 * gap
    x = (width - total) // 2
    y0 = (height - block) // 2
    y1 = y0 + block
    img[y0:y1, x:x + block] = RED
    x += block + gap
    img[y0:y1, x:x + block] = BLUE
    x += block + gap
    mid = y0 + block // 2
    img[y0:mid, x:x + block] = RED
    img[mid:y1, x:x + block] = BLUE
    return img

def chessboard(cols=9, rows=6, square=120, margin=140):
    w = (cols + 1) * square + 2 * margin
    h = (rows + 1) * square + 2 * margin
    img = np.full((h, w, 3), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y = margin + r * square
                x = margin + c * square
                img[y:y + square, x:x + square] = BLACK
    return img

def main():
    os.makedirs(OUT, exist_ok=True)
    print(f'输出目录：{OUT}\n')
    print('【A】识别靶图（拍这些）')
    save(single_block('red'), 'target_red.png', '单个纯红正方形')
    save(single_block('blue'), 'target_blue.png', '单个纯蓝正方形')
    save(single_block('red_blue_h'), 'target_red_blue_h.png', '红蓝相间：上红下蓝')
    save(single_block('red_blue_v'), 'target_red_blue_v.png', '红蓝相间：左红右蓝（立方体两个相邻的面）')
    save(all_three(), 'target_all3.png', '三个物块并排（整体演示用）')
    print('\n【B】相机标定用（拍这个来求 fx/fy/cx/cy）')
    save(chessboard(), 'chessboard_9x6.png', '棋盘格 9x6 内角点（10x7 方格）')
    print(f'\n完成。全部在：{OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
