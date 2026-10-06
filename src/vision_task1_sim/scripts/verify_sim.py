#!/usr/bin/env python3
import sys
import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from vision_task1_interfaces.msg import BlockInfo
CAMERA_MODEL = 'cam'
BLOCKS = {'block_red': 'red', 'block_blue': 'blue', 'block_red_blue': 'red_blue'}
CLASSES = ('red', 'blue', 'red_blue')

class SimVerifier(Node):

    def __init__(self):
        super().__init__('sim_verifier')
        self.truth = {}
        self.errors = {}
        self.n_states = 0
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_model_states, 10)
        self.create_subscription(BlockInfo, '/block_info', self.on_block_info, 50)
        self.create_timer(5.0, self.report)
        self.get_logger().info('等 /gazebo/model_states（真值）和 /block_info（识别结果）...')

    def on_model_states(self, msg: ModelStates):
        idx = {name: i for (i, name) in enumerate(msg.name)}
        if CAMERA_MODEL not in idx:
            return
        self.n_states += 1
        cam = msg.pose[idx[CAMERA_MODEL]].position
        for (model_name, cls) in BLOCKS.items():
            if model_name not in idx:
                continue
            b = msg.pose[idx[model_name]].position
            (dx, dy, dz) = (b.x - cam.x, b.y - cam.y, b.z - cam.z)
            X = -dy
            Y = -dz
            Z = dx
            self.truth[cls] = (X, Y, Z)

    def on_block_info(self, msg: BlockInfo):
        cls = msg.block_class
        if cls not in CLASSES or cls not in self.truth:
            return
        (tx, ty, tz) = self.truth[cls]
        self.errors.setdefault(cls, []).append((msg.x_m - tx, msg.y_m - ty, msg.z_m - tz))

    def report(self):
        if self.n_states == 0:
            self.get_logger().warn('还没收到 /gazebo/model_states —— 仿真起来了吗？')
            return
        if not self.errors:
            self.get_logger().warn('收到真值但没收到 /block_info —— 识别节点在跑吗？相机有出图吗？（ros2 topic hz /camera/image_raw）')
            return
        print()
        print('=' * 88)
        print('  Gazebo 虚拟相机 · 定位精度核对（真值来自仿真器物理位姿，与像素无关）')
        print('=' * 88)
        print(f"  {'类别':<10}{'样本':>6}{'平均|X误差|':>12}{'平均|Y误差|':>12}{'平均|Z误差|':>12}{'最大|Z误差|':>12}{'≤3cm':>8}")
        print('-' * 88)
        for cls in CLASSES:
            errs = self.errors.get(cls)
            if not errs:
                print(f"  {cls:<10}{'0':>6}{'—':>12}{'—':>12}{'—':>12}{'—':>12}{'—':>8}")
                continue
            n = len(errs)
            ax = [abs(e[0]) for e in errs]
            ay = [abs(e[1]) for e in errs]
            az = [abs(e[2]) for e in errs]
            ok = sum((1 for e in errs if max(abs(e[0]), abs(e[1]), abs(e[2])) <= 0.03))
            print(f'  {cls:<10}{n:>6}{sum(ax) / n * 100:>11.2f}cm{sum(ay) / n * 100:>11.2f}cm{sum(az) / n * 100:>11.2f}cm{max(az) * 100:>11.2f}cm{ok / n:>7.1%}')
        print('-' * 88)
        print('  说明：')
        print('    · 真值 = 方块与相机在仿真世界里的真实位姿之差（换算到相机光学坐标系）')
        print('    · 误差 = 识别节点输出的 (x_m, y_m, z_m) − 真值')
        print('    · 题目要求：距摄像头中心点坐标的误差 ≤ 3cm')
        print('    · edge_length_m 字段输出的是设定值 real_edge_m（方块本来就是 5cm），')
        print('      所以它不构成精度证据，真正被考核的是上面的 X / Y / Z')
        print()

def main():
    rclpy.init()
    node = SimVerifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
if __name__ == '__main__':
    sys.exit(main())
