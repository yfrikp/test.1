#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题目1 · ROS2 节点版 —— 把识别结果发布到话题 /block_info
=============================================================================
题目原话："能够通过 ros2 话题输出物块的种类、长度以及距离摄像头中心点的坐标"
所以【必须】有这一个文件才算完成。它把 task1_detect.py 的算法结果 publish 出去。

它复用了 task1_detect.py 里的 detect()，不重复写一遍算法 ——
所以你调参只改 task1_detect.py 顶部那个 P 字典，两边同时生效。

-----------------------------------------------------------------------------
★ 60Hz 达标的关键设计（重要，这是最容易翻车的地方）
   识别一帧要几十毫秒（几 Hz~几十 Hz），但题目要话题频率 ≥60Hz。
   如果把"识别"和"发布"写在同一个循环里，话题频率 = 识别频率，必然不达标。
   所以这里把两件事【拆开】：
     · 一个后台线程专门做识别，算完把最新结果放进 self.latest
     · 一个定时器按 republish_hz（默认 60Hz）把最新结果发出去
   识别慢一点没关系，话题频率永远是 60Hz。这就是"识别与发布解耦"。

-----------------------------------------------------------------------------
用法（在 Ubuntu 终端里，先 source 两个环境）：

    # 静态图片模式（推荐先跑通这个，最省事，也最容易达标 60Hz）
    python3 task1_node.py --ros-args \
        -p image_path:=/mnt/c/Users/Lenovo/dsh-workspace/vision_task1/samples/first.png

    # 图像话题模式（接真实相机/仿真相机时用）
    python3 task1_node.py

  另开一个终端验收：
    ros2 topic list                  # 应该能看到 /block_info
    ros2 topic echo /block_info      # 看内容：block_class / edge_length_m / x_m y_m z_m
    ros2 topic hz /block_info        # 看频率：应该 >= 60
=============================================================================
"""

import os
import sys
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from vision_task1_interfaces.msg import BlockInfo

# 让 import 找得到同目录的 task1_detect.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task1_detect as td            # noqa: E402  复用单机版的 detect()


# =============================================================================
#  ★ 为什么这里不用 cv_bridge（实测踩的坑，很重要）
#
#  这个环境里 pip 装的是 opencv 5.0.0（它是针对 numpy 2.x 编译的），
#  而 ROS2 自带的 cv_bridge 是针对 numpy 1.x 编译的。两者混用会报：
#        AttributeError: _ARRAY_API not found
#        KeyError: 16
#  结果：cv_bridge 的 imgmsg_to_cv2 / cv2_to_imgmsg【全都不能用】，
#        接真实相机/仿真相机的话题模式直接废掉。
#
#  解决办法：干脆不用 cv_bridge。ROS2 的 Image 消息本质上就是
#  「原始字节 + 宽 + 高 + 每行步长」，自己转 5 行就完事，还少一个依赖。
#  支持 bgr8 / rgb8 / mono8 / bgra8（普通彩色相机就够了）。
# =============================================================================
def imgmsg_to_numpy(msg: Image):
    """sensor_msgs/Image -> numpy BGR 图像（替代 cv_bridge）"""
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = (msg.encoding or '').lower()

    if enc in ('bgr8', 'rgb8'):
        arr = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        if enc == 'rgb8':
            arr = arr[:, :, ::-1]                 # RGB -> BGR，OpenCV 要 BGR
        return np.ascontiguousarray(arr)

    if enc == 'bgra8':
        arr = buf.reshape(h, step)[:, :w * 4].reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if enc in ('mono8', '8uc1'):
        arr = buf.reshape(h, step)[:, :w].reshape(h, w)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise ValueError(f'暂不支持的图像编码：{msg.encoding}（需要就在这个函数里加一个分支）')


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')

        # ------------------------------------------------------------------
        #  参数：不写死在代码里，这样命令行、yaml 文件都能改
        #  （和 config/params.yaml 里的名字保持一致）
        # ------------------------------------------------------------------
        self.declare_parameter('real_edge_m', td.P['real_edge_m'])
        self.declare_parameter('fx', td.P['fx'])
        self.declare_parameter('fy', td.P['fy'])
        self.declare_parameter('cx', td.P['cx'])
        self.declare_parameter('cy', td.P['cy'])
        self.declare_parameter('offset_x_m', td.P['offset_x_m'])
        self.declare_parameter('offset_y_m', td.P['offset_y_m'])
        self.declare_parameter('image_path', '')      # 非空 -> 静态图模式
        self.declare_parameter('republish_hz', 60.0)  # 话题发布频率（验收用）
        self.declare_parameter('show_image', False)   # 弹窗看图（会拖慢帧率，测频率时关掉）
        self.declare_parameter('frame_id', 'camera_optical_frame')

        # 把 ROS2 参数同步进算法参数区，保证两边一套参数
        for key in ('real_edge_m', 'fx', 'fy', 'cx', 'cy', 'offset_x_m', 'offset_y_m'):
            td.P[key] = float(self.get_parameter(key).value)

        self.frame_id = self.get_parameter('frame_id').value
        self.show_image = bool(self.get_parameter('show_image').value)
        republish_hz = float(self.get_parameter('republish_hz').value)

        self.pub = self.create_publisher(BlockInfo, 'block_info', 10)

        self.latest = []                 # 最新一帧的识别结果
        self.latest_frame = None         # 最新一帧图像（只用于显示）
        self.lock = threading.Lock()     # 保护 latest / latest_frame
        self.new_frame = None            # 待处理的新帧（话题模式用）
        self.new_frame_lock = threading.Lock()

        self.pub_count = 0
        self.det_count = 0
        self.t_last_report = time.time()

        image_path = self.get_parameter('image_path').value

        if image_path:
            # ================= 模式 A：静态图片 =================
            # 题目明说"可以从网上寻找图片并下载"，这是最省事的验证方式。
            # 识别一次，之后定时器反复发，话题频率稳稳 60Hz。
            if not os.path.exists(image_path):
                self.get_logger().error(f'图片不存在：{image_path}')
                raise SystemExit(1)
            img = cv2.imread(image_path)
            if img is None:
                self.get_logger().error(f'图片读不出来：{image_path}')
                raise SystemExit(1)
            t0 = time.time()
            results, _, _ = td.detect(img, td.P)
            dt = time.time() - t0
            with self.lock:
                self.latest = results
                self.latest_frame = img
            self.get_logger().info(
                f'静态图模式：{image_path}  {img.shape[1]}x{img.shape[0]}  '
                f'识别到 {len(results)} 个物块，单帧耗时 {dt * 1000:.1f} ms')
            for r in results:
                self.get_logger().info(
                    f'   {r["kind"]:<10} 边长={td.P["real_edge_m"] * 100:.1f}cm  '
                    f'X={r["x"] * 100:+.2f}cm Y={r["y"] * 100:+.2f}cm Z={r["z"] * 100:.2f}cm  '
                    f'置信={r["conf"]:.1%}')
        else:
            # ================= 模式 B：订阅图像话题 =================
            # QoS 用 qos_profile_sensor_data —— 相机通常发 best_effort，
            # 订阅方如果用默认的 reliable 会【收不到任何数据】（新手最常见的坑）。
            self.sub_img = self.create_subscription(
                Image, 'image_raw', self.on_image, qos_profile_sensor_data)
            self.sub_info = self.create_subscription(
                CameraInfo, 'camera_info', self.on_camera_info, qos_profile_sensor_data)
            # 识别线程：独立于发布，互不阻塞
            self.worker = threading.Thread(target=self.worker_loop, daemon=True)
            self.worker.start()
            self.get_logger().info('话题模式：等待 /image_raw 和 /camera_info ...')

        # ------------------------------------------------------------------
        #  ★ 发布定时器：固定 republish_hz，与识别速度无关
        # ------------------------------------------------------------------
        self.timer = self.create_timer(1.0 / republish_hz, self.publish_tick)
        self.report_timer = self.create_timer(2.0, self.report_rate)
        self.get_logger().info(
            f'发布频率设定为 {republish_hz:.1f} Hz（识别在独立线程里跑）')

    # ------------------------------------------------------------------
    #  图像回调：只把最新帧存起来，不在这里做识别（识别慢，会堵住回调线程）
    # ------------------------------------------------------------------
    def on_image(self, msg: Image):
        try:
            frame = imgmsg_to_numpy(msg)
        except Exception as e:                                   # noqa: BLE001
            self.get_logger().warn(f'图像转换失败：{e}', once=True)
            return
        with self.new_frame_lock:
            self.new_frame = frame

    def on_camera_info(self, msg: CameraInfo):
        """收到相机内参就用真实的替换掉兜底值 —— 这是 3cm 精度的前提。"""
        k = msg.k
        if k[0] > 0 and k[4] > 0:
            td.P['fx'], td.P['fy'] = float(k[0]), float(k[4])
            td.P['cx'], td.P['cy'] = float(k[2]), float(k[5])
            self.get_logger().info(
                f'已从 /camera_info 获取内参：fx={k[0]:.1f} fy={k[4]:.1f} '
                f'cx={k[2]:.1f} cy={k[5]:.1f}', once=True)

    # ------------------------------------------------------------------
    #  识别线程：拿最新帧 -> 算 -> 更新结果
    # ------------------------------------------------------------------
    def worker_loop(self):
        while rclpy.ok():
            frame = None
            with self.new_frame_lock:
                if self.new_frame is not None:
                    frame = self.new_frame
                    self.new_frame = None
            if frame is None:
                time.sleep(0.001)        # 没有新帧就等一下，别空转烧 CPU
                continue
            results, _, _ = td.detect(frame, td.P)
            with self.lock:
                self.latest = results
                self.latest_frame = frame
            self.det_count += 1

    # ------------------------------------------------------------------
    #  定时器：把最新结果发出去（一个物块一条消息）
    # ------------------------------------------------------------------
    def publish_tick(self):
        with self.lock:
            results = list(self.latest)
            frame = self.latest_frame
        if not results:
            return

        stamp = self.get_clock().now().to_msg()
        for r in results:
            m = BlockInfo()
            m.header.stamp = stamp
            m.header.frame_id = self.frame_id
            m.block_class = r['kind']
            m.edge_length_m = float(td.P['real_edge_m'])
            m.x_m = float(r['x'])
            m.y_m = float(r['y'])
            m.z_m = float(r['z'])
            m.u_px = float(r['u'])
            m.v_px = float(r['v'])
            m.edge_length_px = float(r['edge_px'])
            m.confidence = float(r['conf'])
            self.pub.publish(m)
            self.pub_count += 1

        if self.show_image and frame is not None:
            cv2.imshow('block_info', td.draw(frame, results))
            cv2.waitKey(1)

    def report_rate(self):
        """每 2 秒打印一次真实话题频率 —— 这就是验收 60Hz 的证据。"""
        now = time.time()
        dt = now - self.t_last_report
        if dt <= 0:
            return
        hz = self.pub_count / dt
        ok = '达标' if hz >= 60 else '不达标'
        self.get_logger().info(
            f'>>> 话题频率 = {hz:.1f} Hz  (要求 >= 60 Hz, {ok})   '
            f'识别线程已处理 {self.det_count} 帧   目标数 {len(self.latest)}')
        self.pub_count = 0
        self.t_last_report = now


def main():
    rclpy.init()
    try:
        node = ColorDetector()
    except SystemExit:
        rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
