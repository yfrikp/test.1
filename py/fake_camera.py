#!/usr/bin/env python3
import sys, cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
img = cv2.imread(sys.argv[1])
assert img is not None, '图片读不到'

class FakeCam(Node):

    def __init__(self):
        super().__init__('fake_camera')
        self.p = self.create_publisher(Image, 'image_raw', 10)
        (self.h, self.w) = img.shape[:2]
        self.create_timer(1.0 / 30.0, self.tick)

    def tick(self):
        m = Image()
        m.header.stamp = self.get_clock().now().to_msg()
        (m.height, m.width) = (self.h, self.w)
        m.encoding = 'bgr8'
        m.step = self.w * 3
        m.data = img.tobytes()
        self.p.publish(m)
rclpy.init()
n = FakeCam()
print(f'假相机启动：{n.w}x{n.h} @30fps -> /image_raw', flush=True)
rclpy.spin(n)
