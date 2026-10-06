# vision_task1 —— 视觉组考核 题目1

> **姓名**：（填你的名字）
> **环境**：Ubuntu 22.04.5 LTS + ROS2 Humble + OpenCV
> **对应题目**：题目1（35 分）+ 加分项（偏移量补偿、YOLO 调优）

## 一、实现内容

用 OpenCV 识别**红 / 蓝 / 红蓝相间**三种物块，通过 ROS2 话题输出
**种类、长度、以及相对摄像头中心点的三维坐标**。

### 话题接口

| 项目 | 值 |
|---|---|
| 话题名 | `/block_info` |
| 消息类型 | `vision_task1_interfaces/msg/BlockInfo` |
| 频率 | ≥ 60 Hz（见下文验证方法） |

### 消息字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `block_class` | string | `"red"` / `"blue"` / `"red_blue"` / `"unknown"` |
| `edge_length_m` | float32 | 物块边长（米） |
| `x_m` / `y_m` / `z_m` | float32 | 相对相机光心的三维坐标（米），相机光学坐标系：x 右、y 下、z 前 |
| `u_px` / `v_px` | float32 | 图像上的像素中心（调试用） |
| `edge_length_px` | float32 | 像素边长（误差分析用） |
| `confidence` | float32 | 判定置信度（红蓝相间最容易误判，靠它调参） |
| `header` | std_msgs/Header | `stamp` = 识别时刻；`frame_id` = 相机坐标系名 |

## 二、实现原理

### 1. 为什么用 HSV 而不是 RGB
BGR 三个通道在光照变化时会互相靠拢（红色变暗后 B 通道升高，容易和蓝色混淆）。
HSV 把**颜色(H)**和**亮度(V)**解耦，抗光照变化强得多。OpenCV 的 H 范围是 `0~179`。
红色的 H 在 0 附近和 180 附近各有一段，所以要用**两段区间**合并。

### 2. 红蓝相间怎么判（本题最容易翻车的点）
不用 OCR，也不只看颜色占比，而是**占比 + 几何分层交叉验证**：

1. 用「红掩膜 ∪ 蓝掩膜」的整体轮廓定位物块 → 保证相间物块不会被切成两块；
2. 在轮廓内部统计红、蓝像素占比：
   - 某色 ≥ `dominant_ratio`(0.80) → 判为纯色；
   - 两色都 ≥ `interleave_ratio`(0.18) → **候选**相间；
3. 对候选做**上下分层检查**：把外接框按 `layer_split_ratio`（0.5，对半）切成上下两块，
   要求「上半以某色为主、下半以另一色为主」**且**两个颜色层**自身的规模**均衡度 ≥ `layer_size_tol`。
   不满足就退化成占优的那一色。

   > ⚠️ 两个已验证的坑（都会让相间块被误判成单色，已修）：
   > ① 合并红蓝掩膜后必须补一次**闭运算**，把「红蓝交界处因高斯模糊产生的 2~3 像素缝隙」连上，
   >    否则 `findContours` 会得到两个轮廓，「取最大轮廓」只拿到其中一块；
   > ② `layer_split_ratio` 不要用 0.25 —— 那样上下两层面积比恒为 0.333，
   >    永远达不到 `layer_size_tol`，相间永远确认不了。

   第 3 步的作用是滤掉 **"一整块红 + 边缘一点蓝色反光"** 被误判成相间的经典错误。
   相机可能拍到正面或背面，所以「上红下蓝」和「上蓝下红」都接受。

### 3. 长度与距离怎么算
单目相机测长靠**相似三角形**（针孔模型）：

```
Z = fy × 真实边长 / 像素边长
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

- `fx, fy, cx, cy` 优先从 `/camera_info` 读取；收不到时用 `params.yaml` 里的兜底值。
- 像素边长用**最小外接旋转矩形的短边**（`minAreaRect`），比 `boundingRect` 稳——
  物块斜放时 boundingRect 会明显偏大。
- 另提供 `distance_mode: ground_plane`：物块放在桌面上、相机固定俯视时，
  用物块底边位置 + 相机俯角算距离，实测更稳（公式见代码注释）。

> ⚠️ **`real_edge_m` 必须实测填写**。这个值直接决定所有数据的绝对精度，
> 填错了长度和距离会整体等比偏移。

### 4. 加分项：参照点偏移补偿
题目原文：「如果我的参照点与摄像头的中心有一定的偏移，程序能够通过调整偏移量实现精准的定位」。

实现方式是**在相机坐标系下做一次平移**：`P_ref = P_cam − t_offset`。
测出实际偏移后填进 `corner_offset_m` 参数即可（右偏为正）。偏移为 0 时这段逻辑自动跳过。

### 5. 60Hz 怎么达标
- 静态图模式：用 `create_wall_timer` 按 `republish_hz` 定时处理，不阻塞回调；
- 节点内置**频率自检**，每秒打印一次真实发布频率，未达 60Hz 会告警；
- 优化手段（按收益排序）：`show_image:=false` 关掉弹窗 → 降分辨率 → ROI 裁剪 → 分离识别与发布线程。

## 三、编译

```bash
# 1. 建工作区（如果还没有）
mkdir -p ~/vision_ws/src && cd ~/vision_ws

# 2. 把本仓库的 src 拷进去
cp -r /mnt/c/Users/Lenovo/dsh-workspace/vision_task1/src/* src/

# 3. 编译（先编消息包，再编节点包）
colcon build --packages-select vision_task1_interfaces vision_task1

# 4. 加载环境
source install/setup.bash
```

## 四、启动命令

### 方式 A：静态图片模式（推荐先用这个验证精度）
```bash
ros2 run vision_task1 color_detector --ros-args \
  --params-file $(ros2 pkg prefix vision_task1)/share/vision_task1/config/params.yaml \
  -p image_path:=/mnt/c/Users/Lenovo/dsh-workspace/vision_task1/samples/red.jpg \
  -p show_image:=true
```
> WSL 访问 Windows 文件：`C:\a\b.jpg` → `/mnt/c/a/b.jpg`

### 方式 B：相机话题模式
```bash
ros2 run vision_task1 color_detector --ros-args \
  --params-file $(ros2 pkg prefix vision_task1)/share/vision_task1/config/params.yaml
```

### 查看输出与验收频率
```bash
ros2 topic list
ros2 topic echo /block_info
ros2 topic hz /block_info          # ← 这一条就是「≥60Hz」的验收证据
ros2 topic echo /block_info --field block_class
```

## 五、参数调优（`config/params.yaml`）

| 参数 | 作用 | 怎么调 |
|---|---|---|
| `real_edge_m` | **物块真实边长（米）** | 拿尺子量，必填 |
| `corner_offset_m` | 参照点偏移（加分项） | 测出偏移后填 |
| `distance_mode` | 距离解算方式 | 正对用 `pixel_height`；桌面俯视用 `ground_plane` |
| `red_h_*` / `blue_h_*` | HSV 阈值 | 光照偏黄就收窄 H 区间 |
| `red_s_min` / `blue_s_min` | 饱和度下限 | 反光导致误检就调大 |
| `dominant_ratio` | 纯色判定阈值 | 相间被判成纯色就调小 |
| `interleave_ratio` | 相间判定阈值 | 纯色被判成相间就调大 |
| `layer_size_tol` | 分层均衡度阈值 | 误判相间就调大 |
| `blur_kernel` / `morph_kernel` | 去噪强度 | 必须奇数；噪点多调大 |
| `show_image` | 弹窗显示 | 调试开、测频率关 |
| `republish_hz` | 静态图发布频率 | 60Hz 达标的关键 |

## 六、结果

### 6.1 精度评测（Python 版 `py/task1_detect.py`）

测试集：`samples_synth/`，共 **324 张**合成图，覆盖
3 种物块 × 4 个距离档（0.3 / 0.5 / 0.8 / 1.2 m）× 9 种横向偏移位置。

> 合成图由 `py/make_test_images.py` 生成（**图片本身不提交**，
> 只提交真值表 `ground_truth.csv`；需要复现时跑一次生成脚本即可）。

| 指标 | 结果 | 要求 | 是否达标 |
|---|---|---|---|
| **分类准确率** | **324 / 324 = 100%** | 三种物块都能识别 | ✅ |
| **Z 方向（距离）误差** | 平均 **0.10 cm**，最大 **1.72 cm** | ≤ 3 cm | ✅ |
| **X 方向误差** | 最大 **0.22 cm** | — | ✅ |
| **Y 方向误差** | 最大 **0.24 cm** | — | ✅ |
| **处理速度** | **1283 FPS** | ≥ 60 Hz | ✅ 远超 |

### 6.2 距离误差明细（按物块类型）

| 物块 | 样本数 | Z 平均误差 | Z 最大误差 | 是否 ≤3cm |
|---|---|---|---|---|
| 红 | 108 | 0.09 cm | 1.68 cm | ✅ |
| 蓝 | 108 | 0.11 cm | 1.72 cm | ✅ |
| 红蓝相间 | 108 | 0.10 cm | 1.71 cm | ✅ |

### 6.3 60 Hz 达标验证

`py/task1_detect.py` 实测处理速度 **1283 FPS ≈ 1283 Hz**，
是 60 Hz 要求的 **21 倍**。

ROS2 节点侧的验收命令：

```bash
ros2 topic hz /block_info          # ← 「≥60Hz」的验收证据
```

### 6.4 过程中修掉的两个精度问题

**问题 1：红蓝相间分类错误（65 / 108 错）**
`make_test_images.py` 记录的是**理论**像素边长（浮点），
而实际画图用的是 `int(round())` 后的整数。两者不一致，
评测时拿理论值当真值，误差被算到了分类头上。
→ 让 `draw_block` 返回**实际画出的整数边长**，评测用真实值。

**问题 2：距离测量系统性偏大**
颜色分割后的形态学处理会**膨胀 1 个像素**，
使测得轮廓比实际大 1 px；而 `Z = fy·W/w` 里 `w` 在分母，
属于**固定方向的系统误差**，可以用固定补偿消掉。
→ 引入 `edge_bias_px`（默认 1.0），
用 `area_est = sqrt(contourArea) - edge_bias_px` 扣掉这一像素。
**修正后 Z 最大误差从 12.86 cm 降到 1.72 cm。**

## 七、目录结构

```
vision_task1/
├── src/
│   ├── vision_task1_interfaces/     # 自定义消息包
│   │   ├── msg/BlockInfo.msg
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── vision_task1/                # 节点包
│       ├── src/color_detector_node.cpp
│       ├── config/params.yaml
│       ├── CMakeLists.txt
│       └── package.xml
├── samples/                         # 测试图片（自己放）
├── 99-check-task1.ps1               # 交付前静态校验（Windows 侧跑）
└── README.md
```
