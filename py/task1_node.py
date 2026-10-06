#!/usr/bin/env python3
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task1_detect as td

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
    raise ValueError(f'暂不支持的图像编码：{msg.encoding}（需要就在这个函数里加一个分支）')

class ColorDetector(Node):

    def __init__(self):
        super().__init__('color_detector')
        self.declare_parameter('real_edge_m', td.P['real_edge_m'])
        self.declare_parameter('fx', td.P['fx'])
        self.declare_parameter('fy', td.P['fy'])
        self.declare_parameter('cx', td.P['cx'])
        self.declare_parameter('cy', td.P['cy'])
        self.declare_parameter('offset_x_m', td.P['offset_x_m'])
        self.declare_parameter('offset_y_m', td.P['offset_y_m'])
        self.declare_parameter('image_path', '')
        self.declare_parameter('republish_hz', 60.0)
        self.declare_parameter('show_image', False)
        self.declare_parameter('frame_id', 'camera_optical_frame')
        for key in ('real_edge_m', 'fx', 'fy', 'cx', 'cy', 'offset_x_m', 'offset_y_m'):
            td.P[key] = float(self.get_parameter(key).value)
        self.frame_id = self.get_parameter('frame_id').value
        self.show_image = bool(self.get_parameter('show_image').value)
        republish_hz = float(self.get_parameter('republish_hz').value)
        self.pub = self.create_publisher(BlockInfo, 'block_info', 10)
        self.latest = []
        self.latest_frame = None
        self.lock = threading.Lock()
        self.new_frame = None
        self.new_frame_lock = threading.Lock()
        self.pub_count = 0
        self.det_count = 0
        self.t_last_report = time.time()
        image_path = self.get_parameter('image_path').value
        if image_path:
            if not os.path.exists(image_path):
                self.get_logger().error(f'图片不存在：{image_path}')
                raise SystemExit(1)
            img = cv2.imread(image_path)
            if img is None:
                self.get_logger().error(f'图片读不出来：{image_path}')
                raise SystemExit(1)
            t0 = time.time()
            (results, _, _) = td.detect(img, td.P)
            dt = time.time() - t0
            with self.lock:
                self.latest = results
                self.latest_frame = img
            self.get_logger().info(f'静态图模式：{image_path}  {img.shape[1]}x{img.shape[0]}  识别到 {len(results)} 个物块，单帧耗时 {dt * 1000:.1f} ms')
            for r in results:
                self.get_logger().info(f"   {r['kind']:<10} 边长={td.P['real_edge_m'] * 100:.1f}cm  X={r['x'] * 100:+.2f}cm Y={r['y'] * 100:+.2f}cm Z={r['z'] * 100:.2f}cm  置信={r['conf']:.1%}")
        else:
            self.sub_img = self.create_subscription(Image, 'image_raw', self.on_image, qos_profile_sensor_data)
            self.sub_info = self.create_subscription(CameraInfo, 'camera_info', self.on_camera_info, qos_profile_sensor_data)
            self.worker = threading.Thread(target=self.worker_loop, daemon=True)
            self.worker.start()
            self.get_logger().info('话题模式：等待 /image_raw 和 /camera_info ...')
        self.timer = self.create_timer(1.0 / republish_hz, self.publish_tick)
        self.report_timer = self.create_timer(2.0, self.report_rate)
        self.get_logger().info(f'发布频率设定为 {republish_hz:.1f} Hz（识别在独立线程里跑）')

    def on_image(self, msg: Image):
        try:
            frame = imgmsg_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f'图像转换失败：{e}', once=True)
            return
        with self.new_frame_lock:
            self.new_frame = frame

    def on_camera_info(self, msg: CameraInfo):
        k = msg.k
        if k[0] > 0 and k[4] > 0:
            (td.P['fx'], td.P['fy']) = (float(k[0]), float(k[4]))
            (td.P['cx'], td.P['cy']) = (float(k[2]), float(k[5]))
            self.get_logger().info(f'已从 /camera_info 获取内参：fx={k[0]:.1f} fy={k[4]:.1f} cx={k[2]:.1f} cy={k[5]:.1f}', once=True)

    def worker_loop(self):
        while rclpy.ok():
            frame = None
            with self.new_frame_lock:
                if self.new_frame is not None:
                    frame = self.new_frame
                    self.new_frame = None
            if frame is None:
                time.sleep(0.001)
                continue
            (results, _, _) = td.detect(frame, td.P)
            with self.lock:
                self.latest = results
                self.latest_frame = frame
            self.det_count += 1

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
        now = time.time()
        dt = now - self.t_last_report
        if dt <= 0:
            return
        hz = self.pub_count / dt
        ok = '达标' if hz >= 60 else '不达标'
        self.get_logger().info(f'>>> 话题频率 = {hz:.1f} Hz  (要求 >= 60 Hz, {ok})   识别线程已处理 {self.det_count} 帧   目标数 {len(self.latest)}')
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
