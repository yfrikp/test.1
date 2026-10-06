#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

def imgmsg_to_numpy(msg: Image):
    (h, w, step) = (msg.height, msg.width, msg.step)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = (msg.encoding or '').lower()
    if enc in ('bgr8', 'rgb8'):
        arr = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        if enc == 'rgb8':
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == 'bgra8':
        arr = buf.reshape(h, step)[:, :w * 4].reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if enc in ('mono8', '8uc1'):
        arr = buf.reshape(h, step)[:, :w].reshape(h, w)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    raise ValueError(f'暂不支持的编码：{msg.encoding}')

class FrameSaver(Node):

    def __init__(self, topic, out, hsv_report):
        super().__init__('save_frame')
        self.out = out
        self.hsv_report = hsv_report
        self.done = False
        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f'等 {topic} 的第一帧 ...')

    def on_image(self, msg: Image):
        if self.done:
            return
        img = imgmsg_to_numpy(msg)
        cv2.imwrite(self.out, img)
        self.get_logger().info(f'已保存 {self.out}   {img.shape[1]}x{img.shape[0]}   编码={msg.encoding}')
        if self.hsv_report:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            (h, s, v) = (hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2])
            m = (s > 80) & (v > 60)
            if m.sum() > 0:
                self.get_logger().info(f'有颜色像素 {int(m.sum())} 个：H中位={int(np.median(h[m]))}  S中位={int(np.median(s[m]))}  V中位={int(np.median(v[m]))}')
                red = m & ((h <= 12) | (h >= 168))
                blue = m & (h >= 95) & (h <= 140)
                self.get_logger().info(f'  其中偏红 {int(red.sum())} 个，偏蓝 {int(blue.sum())} 个')
            else:
                self.get_logger().warn('画面里没有检测到饱和颜色 —— 方块在视野里吗？')
        self.done = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/frame.png')
    ap.add_argument('--topic', default='/camera/image_raw')
    ap.add_argument('--hsv', action='store_true', help='顺便打印 HSV 统计')
    args = ap.parse_args()
    rclpy.init()
    node = FrameSaver(args.topic, args.out, args.hsv)
    try:
        while rclpy.ok() and (not node.done):
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.done else 1
if __name__ == '__main__':
    sys.exit(main())
