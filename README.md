# 题目1 · 颜色物块识别与定位

> 用 OpenCV 识别**红 / 蓝 / 红蓝相间**三种物块，通过 **ROS2 话题**输出
> **种类 + 长度 + 距摄像头中心点的坐标**，误差 **≤3cm**，频率 **≥60Hz**。

**这是一个独立工作区**：只放题目1 的东西，和题目2（`~/vision_ws` / `vision_task2`）完全分开，
怎么折腾都不会影响第二题。

---

## 一、30 秒看现状

| 题目要求 | 状态 | 证据在哪 |
|---|---|---|
| 识别三种物块 | ✅ **324/324 = 100%** | [docs/验收证据.md](docs/验收证据.md) |
| 输出**种类** | ✅ `block_class` 字段 | `ros2 topic echo /block_info` |
| 输出**长度** | ✅ `edge_length_m`（米） | 同上 |
| 输出**距摄像头中心坐标** | ✅ `x_m / y_m / z_m`（米） | 同上 |
| 通过 **ROS2 话题**输出 | ✅ `/block_info` | 同上 |
| 误差 **≤3cm** | ✅ 合成图集 **100%**；仿真 **100%**（每类 710 样本） | [docs/验收证据.md](docs/验收证据.md) |
| 频率 **≥60Hz** | ✅ 实测 **180 Hz** | `ros2 topic hz /block_info` |
| 加分项① 光心偏移补偿 | ✅ 参数 `offset_x_m/offset_y_m` | `py/task1_detect.py` |
| 加分项② YOLO 调优 | ⬜ 未做（可选） | —— |

---

## 二、快速开始

打开终端（VS Code 里 `Ctrl+`` `），或者按 `Ctrl+Shift+B` 直接跑任务。

```bash
# 1) 编译
colcon build --symlink-install

# 2) 跑一次"无相机"精度验证：Gazebo 虚拟相机 + 识别节点
source install/setup.bash
ros2 launch vision_task1_sim camera_scene.launch.py
```

另开一个终端看结果：

```bash
source install/setup.bash
ros2 topic echo /camera/camera_info --once   # 相机内参（自动给的，不用标定）
ros2 topic echo /block_info                  # 识别结果
ros2 topic hz   /block_info                  # 话题频率（验收 60Hz）
python3 src/vision_task1_sim/scripts/verify_sim.py   # 独立真值核对
```

不想开仿真？直接拿图片跑：

```bash
python3 py/task1_detect.py --src samples/first.png --show
python3 py/eval_synth.py --samples samples_synth        # 324 张带真值的统计
```

### VS Code 任务（推荐用这个）

`Ctrl+Shift+P` → `Tasks: Run Task`：

| 任务 | 作用 |
|---|---|
| ① 编译工作区 | `colcon build --symlink-install` |
| ② 起 Gazebo 虚拟相机场景 | 仿真 + 识别节点一起起 |
| ③ 独立真值核对 | 仿真精度表格 |
| ④ 跑合成图集评测 | 324 张的分类/精度统计 |
| ⑤ 重新生成合成图集 | 132MB 的图集随时重建 |
| ⑥ 生成拍摄靶图 | 含标定棋盘格 |
| ⑦ 单张图片识别 | 会弹框让你填路径 |
| ⑧ 抓一帧相机画面 | 存到 `results/` |
| ⑨ / ⑩ 看话题频率 / 内容 | 验收 60Hz 用 |
| ⑪ 刷新调试环境变量 | 改完环境后跑一次，F5 调试才正常 |

---

## 三、目录结构

```
task1_ws/
├── src/
│   ├── vision_task1_interfaces/   题目1 的自定义消息 BlockInfo.msg（10 个字段）
│   ├── vision_task1/              C++ 节点（另一种交付形态，参数在 config/params.yaml）
│   └── vision_task1_sim/          Gazebo 虚拟相机场景（不用真实相机！）
│       ├── worlds/block_scene.world    相机 + 光源 + 真值 state 插件
│       ├── models/block_*.sdf          三个方块模型
│       ├── launch/camera_scene.launch.py
│       └── scripts/verify_sim.py       独立真值核对 / save_frame.py 抓帧
├── py/                            纯 Python 工具（不依赖 ROS2 也能跑）
│   ├── task1_detect.py            ★ 核心算法（单机版：图片/视频/摄像头）
│   ├── task1_node.py              ★ ROS2 节点（订阅相机、发布 /block_info）
│   ├── eval_synth.py              拿带真值的图集算误差
│   ├── make_test_images.py        生成 324 张带真值的合成图
│   ├── make_targets.py            生成拍摄靶图 + 标定棋盘格
│   ├── tune_hsv.py                拖滑条调 HSV 阈值
│   ├── fake_camera.py             假相机：把图片当视频发到 /image_raw
│   └── selftest.py / learn_opencv.py / detector.py   早期版本与教学脚本
├── targets/                       拍摄靶图（红/蓝/红蓝/棋盘格）+ 拍摄说明
├── samples/                       测试图（含一张手机实拍照片）
├── samples_synth/                 324 张合成图 + ground_truth.csv（132MB，可重建）
├── results/                       ★ 验收证据：评测报告、相机画面
├── docs/                          说明书与验收证据
└── .vscode/                       VS Code 配置（任务、调试、推荐扩展）
```

---

## 四、两条验证路线（都不需要真实相机）

| | ① 合成图集 | ② Gazebo 虚拟相机 |
|---|---|---|
| 怎么跑 | `eval_synth.py` | `camera_scene.launch.py` + `verify_sim.py` |
| 真值来源 | 生成时就记录（同一个针孔模型） | **仿真器物理位姿（完全独立）** |
| 链路真实性 | 读图片文件 | **真话题：相机 → 节点 → 输出** |
| 强项 | 样本多（324 张）、可覆盖多距离多光照 | 真值独立、能测真话题频率 |

**两条都跑通，就等于"算法对 + 系统对"都有了证据。**

---

## 五、四个必须知道的坑（都踩过）

1. **`cv_bridge` 在本机不可用** —— ROS2 的 cv_bridge 针对 numpy 1.x 编译，而 pip 装的
   opencv 5.0 是 numpy 2.x，导入就报 `_ARRAY_API not found` / `KeyError: 16`。
   **代码里一律不用 cv_bridge**（`task1_node.py` 和 `save_frame.py` 里自己转 5 行）。

2. **URDF/图像里的中文不能进 DDS** —— 实测 `robot_description` 带中文注释会让
   `spawn_entity` 失败。`expand_urdf.py` 会自动过滤非 ASCII。

3. **WSL 上 Gazebo 相机必须软件渲染** —— 硬件渲染会 `D3D12: Removing Device.` 然后
   gzserver 段错误（exit -11）。launch 里已默认 `LIBGL_ALWAYS_SOFTWARE=1`。
   代价：仿真相机只有约 12 fps（软件光栅化的上限，不是算法问题）。

4. **形态学 CLOSE 必须作用在"合并后的掩膜"上** —— 对每种颜色各做一次是**补不上缝**的，
   红蓝相间的物块会被拆成"纯红 + 纯蓝"两块，类别判错、距离偏 30%。
   这个 bug 是拿 324 张带真值的图集才发现的。

---

## 六、提交清单（GitHub）

- [x] 代码：`py/task1_detect.py`、`py/task1_node.py`
- [x] 消息定义：`src/vision_task1_interfaces/msg/BlockInfo.msg`
- [x] C++ 版本：`src/vision_task1/`
- [x] 验收证据：`results/合成图集评测结果.txt`、`results/sim_camera_view.png`
- [x] 说明文档：本文件 + `docs/`
- [ ] README 里补一段**你自己拍的实物照片**结果（可选，但最有说服力）
- [ ] 加分项② YOLO 调优（可选）

`.gitignore` 已经把编译产物、132MB 合成图集、机器相关的 `ros.env` 排除掉了，
直接 `git add .` 不会把垃圾带进仓库。
