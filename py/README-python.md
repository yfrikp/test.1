# 题目1 算法开发套件（Windows · Python · 不需要 ROS2）

## 为什么要单独做这一套

你的环境（WSL + ROS2）还在装。但**调算法根本不需要 ROS2**——
HSV 阈值、红蓝相间判定、距离换算是纯 OpenCV 的事。

所以这一套让你**现在就能开始**，而且解决了一个很实际的困难：

> **题目要求"实际误差不能超过 3cm"。要验证精度，你必须知道真实值是多少。**
> 可你现在既没相机也没物块，更没法把物块精确摆到指定距离。

`make_test_images.py` 用代码合成图片：**物块的位置、尺寸、距离都是我设定的**，
于是"标准答案"精确已知，可以直接算误差。

---

## 快速开始

**前置**：装一个真正的 Python 3（你机器上只有微软商店的 0 字节占位符，等于没装）
- 官网 <https://www.python.org/downloads/windows/> 下载安装，**务必勾选 Add python.exe to PATH**
- 或用微软商店搜 "Python 3.12"

然后一键跑：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Lenovo\dsh-workspace\vision_task1\py\run_all.ps1"
```

它会自动：检查环境 → 装依赖（走清华源）→ 生成合成图 → 跑离线自测 → 给出调参指引。

---

## 四个文件

| 文件 | 作用 |
|---|---|
| `detector.py` | **算法本体**。与 C++ 版 `color_detector_node.cpp` 逐行对应 |
| `make_test_images.py` | 生成带精确真值的合成测试图 |
| `selftest.py` | 离线自测：分类准确率 / 距离误差 / 处理速度 |
| `tune_hsv.py` | 交互式 HSV 调参工具（带滑条，实时看掩膜） |

---

## 手动用法

```powershell
cd C:\Users\Lenovo\dsh-workspace\vision_task1\py

# 生成合成测试图（默认 3 类 × 4 距离 × 3×3 偏移 × 3 光照）
python make_test_images.py --out samples

# 也可以自定义
python make_test_images.py --out samples --distances 0.2 0.3 0.5 1.0 --noise 8 --lights 0.6 1.0 1.4

# 跑离线自测
python selftest.py --samples samples

# 交互式调参（--image 可以传文件、目录或通配符）
python tune_hsv.py --image samples
python tune_hsv.py --image real_photo.jpg --edge 0.05
```

调参窗口按键：`s` 保存参数 · `d` 切换掩膜显示 · `n`/`p` 换图 · `q` 退出

---

## 自测报告怎么看

```
1) CLASSIFICATION          分类准确率 + 混淆矩阵
2) ACCURACY                距离/坐标误差统计，对照「≤3cm」
3) SPEED                   单帧耗时与理论帧率，对照「≥60Hz」
```

**合成图上的误差应当接近 0**（因为图和算法用同一套针孔模型）。
这一点很关键，它帮你把两类问题分开：

| 现象 | 说明 |
|---|---|
| 合成图上误差就很大 | **算法或公式有 bug** —— 先解决它 |
| 合成图接近 0、真实照片误差大 | **相机内参不准 / `real_edge_m` 没实测** —— 去标定和量尺寸 |
| 分类准确率 < 100% | HSV 阈值或分层判据需要调 |

---

## 合成图是怎么造的（可验证性）

对每张图，代码显式记录了真值：

```
edge_px = fx * real_edge_m / Z          # 相似三角形
u, v    = project(X, Y, Z)              # 针孔模型投影
```

也就是说：**图上物块的像素边长和位置，是用同一套公式正推出来的**。
`detector.py` 再做反推。正推/反推一致，误差就该是 0——
如果实测不是 0，那就是代码里某处写错了。这就是它能当"单元测试"用的原因。

噪声、光照、JPEG 压缩都是可调的（`--noise` / `--lights` / `--jpeg-quality`），
用来测试阈值在**非理想条件**下还稳不稳。

---

## 与 C++ 版的关系

| 项目 | Python 版（本目录） | C++ 版（提交用） |
|---|---|---|
| 算法逻辑 | 完全一致 | 完全一致 |
| 运行环境 | Windows / 任意有 OpenCV 的地方 | Ubuntu 22.04 + ROS2 Humble |
| 输出 | print + 窗口 | `/block_info` 话题 |
| 频率 | 较慢（解释执行） | **满足 ≥60Hz**（编译执行） |
| 用途 | **提前调通算法与阈值** | **最终提交、验收** |

在 `tune_hsv.py` 里按 `s` 保存的 `params_tuned.yaml`，
结构和 C++ 包的 `config/params.yaml` 一样，**可以直接复制过去**。

---

## 一个已经修掉的坑（值得你知道）

原先距离用 `minAreaRect` 的**短边**算。这是错的：
物块是立方体，正对相机时投影是正方形，侧视时投影变矩形——
**长边仍然对应棱长**（透视缩短只影响进深方向，表现为短边变短）。
用短边会把距离算得偏大。现已改成取**长边**，C++ 版同步修正。

这类错误在合成图上很容易暴露（误差突然变得很大），
而用真实照片则很容易被误以为是"没标定好"。这也是先做合成自测的价值。
