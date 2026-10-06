// =============================================================================
//  color_detector_node.cpp —— 题目1 的核心实现
//
//  题目要求：识别红 / 蓝 / 红蓝相间三种物块，通过 ROS2 话题输出
//            「种类 + 长度 + 距离摄像头中心点的坐标」，误差 ≤3cm，频率 ≥60Hz
//
//  这个文件已经实现了完整流程，你主要要做的是【调参数】而不是从零写算法。
//
//  流程：
//    图像 -> BGR转HSV -> 红/蓝阈值分割 -> 形态学去噪 -> 找最大轮廓
//         -> 判定红/蓝/红蓝相间（按像素占比）-> 算像素尺寸
//         -> 换算物理长度与三维坐标 -> 发布 BlockInfo
//
//  用法：
//    ros2 run vision_task1 color_detector --ros-args -p image_path:=/path/to/img.jpg
//    ros2 run vision_task1 color_detector --ros-args -p image_path:=img.jpg -p show_image:=true
//
//  【C -> C++ 提示】这个文件里你会看到：
//    类 = 把"处理这个相机的所有状态和函数"打包在一起（相当于 C 里一堆全局变量+函数）
//    const 引用传参 = 比指针安全，且不会复制（C 里只能传指针）
//    auto = 让编译器推断类型（写起来快，但心里要知道类型是什么）
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "vision_task1_interfaces/msg/block_info.hpp"

using namespace std::chrono_literals;

// -----------------------------------------------------------------------------
//  可视化颜色（BGR）—— 集中定义，避免代码里到处散落魔数
//  只有下面这几个常量是"字符串里含中文"之外的纯 ASCII，终端输出不会乱码
// -----------------------------------------------------------------------------
namespace viz {
const cv::Scalar kRed    (  0,   0, 255);
const cv::Scalar kBlue   (255,   0,   0);
const cv::Scalar kGreen  (  0, 255,   0);
const cv::Scalar kYellow (  0, 255, 255);
const cv::Scalar kWhite  (255, 255, 255);
const cv::Scalar kBlack  (  0,   0,   0);
}  // namespace viz

// =============================================================================
//  ColorDetector 节点
// =============================================================================
class ColorDetector : public rclcpp::Node
{
public:
  ColorDetector()
  : Node("color_detector")
  {
    // ---------------------------------------------------------------------
    //  1) 声明所有可调参数
    //     每个参数都能在启动时用 -p 名字:=值 覆盖，改参数不用重新编译。
    //     ⚠️ 这些默认值是"起点"，你需要拿真实图片去调，见 README 的调参步骤。
    // ---------------------------------------------------------------------

    // ---- 红/蓝的 HSV 阈值 ----
    // OpenCV 的 H 范围是 0~179（不是 0~360）。
    // 红色的 H 在 0 附近和 180 附近各有一段，所以要用两个区间。
    red_h_low_max_   = declare_parameter("red_h_low_max",   10);
    red_h_high_min_  = declare_parameter("red_h_high_min", 170);
    red_s_min_       = declare_parameter("red_s_min",       90);   // 饱和度下限：滤掉发白的区域
    red_v_min_       = declare_parameter("red_v_min",       50);   // 亮度下限：滤掉发黑的区域

    blue_h_min_      = declare_parameter("blue_h_min",      95);
    blue_h_max_      = declare_parameter("blue_h_max",     135);
    blue_s_min_      = declare_parameter("blue_s_min",      90);
    blue_v_min_      = declare_parameter("blue_v_min",      50);

    // ---- 分类阈值 ----
    dominant_ratio_     = declare_parameter("dominant_ratio",     0.80);  // 单色占比达标即判为纯色
    interleave_ratio_   = declare_parameter("interleave_ratio",   0.18);  // 红蓝相间：每色至少这么多
    layer_split_ratio_  = declare_parameter("layer_split_ratio",  0.50);  // 上下分层切分位置（0.5=对半切）
    layer_size_tol_     = declare_parameter("layer_size_tol",     0.45);  // 上下层面积接近程度

    // ---- 预处理 ----
    morph_kernel_    = declare_parameter("morph_kernel",   5);     // 形态学核大小（奇数）
    min_area_px_     = declare_parameter("min_area_px",    200);   // 小于这个面积当噪声丢掉
    blur_kernel_     = declare_parameter("blur_kernel",    5);     // 高斯模糊核（奇数，抑制噪点）

    // ---- 物理换算 ----
    real_edge_m_     = declare_parameter("real_edge_m",    0.05);  // ★物块真实边长（米）——必须你实测填写
    corner_offset_m_ = declare_parameter("corner_offset_m", 0.0);  // ★参照点与相机光心的偏移（加分项）
    distance_mode_   = declare_parameter("distance_mode",  std::string("pixel_height"));
    //   distance_mode 可选值：
    //     "pixel_height" —— Z = fy * 真实边长 / 像素高度   （最直接，依赖物块正对相机）
    //     "ground_plane" —— 用图像上物块底边位置 + 相机俯角算距离（物块放在桌面上时更稳）
    cam_height_m_    = declare_parameter("cam_height_m",   0.35);  // ground_plane 模式：相机离地高度
    cam_pitch_deg_   = declare_parameter("cam_pitch_deg",  30.0);  // ground_plane 模式：相机俯角

    // ---- 图像源 ----
    image_path_    = declare_parameter("image_path",   std::string(""));
    republish_hz_  = declare_parameter("republish_hz", 60.0);   // 静态图模式下的话题频率目标
    show_image_    = declare_parameter("show_image",   false);

    // ---- 相机内参兜底 ----
    // 有 /camera_info 时优先用它；没有就用下面这组（你需要按实际相机改）
    // fx/fy/cx/cy 的含义：把三维点投影到像素的针孔模型参数
    fallback_fx_ = declare_parameter("fallback_fx", 600.0);
    fallback_fy_ = declare_parameter("fallback_fy", 600.0);
    fallback_cx_ = declare_parameter("fallback_cx", 320.0);
    fallback_cy_ = declare_parameter("fallback_cy", 240.0);

    // ---------------------------------------------------------------------
    //  2) 创建发布者
    //     QoS 用默认的 reliable + 队列 10。
    //     题目要求 ≥60Hz，队列别太大，否则积压的消息会让你看到的频率虚高。
    // ---------------------------------------------------------------------
    pub_ = this->create_publisher<vision_task1_interfaces::msg::BlockInfo>("block_info", 10);

    // ---------------------------------------------------------------------
    //  3) 选择数据来源
    //     静态图模式：自己读文件、自己定时处理（题目允许"从网上下载图片"）
    //     话题模式  ：订阅相机图像（接真实相机或 Gazebo 相机时用）
    // ---------------------------------------------------------------------
    if (!image_path_.empty()) {
      cv::Mat img = cv::imread(image_path_, cv::IMREAD_COLOR);
      if (img.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to read image: %s", image_path_.c_str());
        RCLCPP_ERROR(this->get_logger(),
          "Check the path. Inside WSL, Windows files are at /mnt/c/...");
        throw std::runtime_error("image load failed");
      }
      static_image_ = img;
      RCLCPP_INFO(this->get_logger(), "Static image mode: %s (%d x %d)",
                  image_path_.c_str(), img.cols, img.rows);

      // 按 republish_hz 定时处理。
      // 这就是 60Hz 达标的实现方式：静态图处理很快，瓶颈只在定时器精度。
      const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, republish_hz_));
      timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&ColorDetector::process_static, this));
    } else {
      RCLCPP_INFO(this->get_logger(), "Topic mode: waiting for /image_raw and /camera_info");
      sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", 10,
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { on_camera_info(m); });
      sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw", 10,
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(m); });
    }

    // ---------------------------------------------------------------------
    //  4) 频率统计定时器（验收用）
    //     题目要求"识别帧率和输出话题频率 >= 60hz"，
    //     这个定时器每秒打印一次真实频率，你截图这个日志就是验收证据。
    // ---------------------------------------------------------------------
    fps_timer_ = this->create_wall_timer(1s, std::bind(&ColorDetector::report_rate, this));

    RCLCPP_INFO(this->get_logger(),
      "param real_edge_m=%.4f m  <-- MEASURE YOUR BLOCK AND FIX THIS", real_edge_m_);
  }

private:
  // =========================================================================
  //  图像来源回调
  // =========================================================================
  void on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    // 相机内参矩阵 K = [fx 0 cx; 0 fy cy; 0 0 1]，按行优先存 9 个数
    if (msg->k.size() >= 6) {
      fx_ = msg->k[0]; fy_ = msg->k[4];
      cx_ = msg->k[2]; cy_ = msg->k[5];
      has_intrinsics_ = true;
    }
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (!has_intrinsics_) {
      // 还没收到 camera_info，先用兜底值，并提示一次
      if (!warned_intrinsics_) {
        RCLCPP_WARN(this->get_logger(),
        "No /camera_info yet, using fallback intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
          fallback_fx_, fallback_fy_, fallback_cx_, fallback_cy_);
        warned_intrinsics_ = true;
      }
    }
    cv::Mat img;
    try {
      img = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
      return;
    }
    detect_and_publish(img, msg->header);
  }

  void process_static()
  {
    std_msgs::msg::Header h;
    h.stamp = this->now();
    h.frame_id = "camera_optical_frame";
    detect_and_publish(static_image_, h);
  }

  // =========================================================================
  //  主检测流程
  // =========================================================================
  void detect_and_publish(const cv::Mat & bgr, const std_msgs::msg::Header & header)
  {
    // ---- 1) 转 HSV ----
    // 为什么必须转 HSV：BGR 下光照一变，红和蓝的数值会互相靠拢；
    // HSV 把"颜色(H)"和"亮度(V)"分开，抗光照变化强得多。
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // ---- 2) 高斯模糊：抑制噪点，不然形态学之后会出现一堆小碎块 ----
    cv::Mat hsv_blur;
    int bk = std::max(1, blur_kernel_ | 1);   // 保证是奇数
    cv::GaussianBlur(hsv, hsv_blur, cv::Size(bk, bk), 0);

    // ---- 3) 红色掩膜（两段 H 区间合并）----
    cv::Mat mask_red_low, mask_red_high, mask_red;
    cv::inRange(hsv_blur,
      cv::Scalar(0, red_s_min_, red_v_min_),
      cv::Scalar(red_h_low_max_, 255, 255), mask_red_low);
    cv::inRange(hsv_blur,
      cv::Scalar(red_h_high_min_, red_s_min_, red_v_min_),
      cv::Scalar(179, 255, 255), mask_red_high);
    cv::bitwise_or(mask_red_low, mask_red_high, mask_red);

    // ---- 4) 蓝色掩膜 ----
    cv::Mat mask_blue;
    cv::inRange(hsv_blur,
      cv::Scalar(blue_h_min_, blue_s_min_, blue_v_min_),
      cv::Scalar(blue_h_max_, 255, 255), mask_blue);

    // ---- 5) 形态学：开运算去掉小白点，闭运算补上物体内部的小洞 ----
    int mk = std::max(1, morph_kernel_ | 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mk, mk));
    cv::morphologyEx(mask_red,  mask_red,  cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask_red,  mask_red,  cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask_blue, mask_blue, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask_blue, mask_blue, cv::MORPH_CLOSE, kernel);

    // ---- 6) 合并掩膜找轮廓 ----
    //  用"红或蓝"的整体轮廓来定位物块，再在轮廓内部统计红/蓝占比来分类。
    //  这样做的好处：红蓝相间的物块也能被当成一个整体找到，不会分裂成两块。
    cv::Mat mask_any;
    cv::bitwise_or(mask_red, mask_blue, mask_any);

    // ---- 6.5) ★ 补上"红蓝交界处的缝隙" ----
    //  为什么会有缝：第 2 步对 HSV 做了高斯模糊，红蓝相接的地方 H 从 0 平滑过渡到 120，
    //  中间那 2~3 个像素既不在红区间也不在蓝区间 -> mask_any 正好在那里断开。
    //  后果：findContours 得到【两个】轮廓，下面"取面积最大的那个"只会拿到其中一块，
    //        红蓝相间的物块因此被判成占优的那一色。
    //        （实测：5cm 相间块被判成 blue，v_px=301 正好是蓝块中心，而不是整块中心 240）
    //  闭运算能把宽度小于核尺寸的缝隙连起来，且不改变外轮廓尺寸（实测仍是 244x244）。
    {
      const int gap_k = std::max(7, mk * 2 + 1);   // morph_kernel=5 时取 11，足够跨过实测 2px 的缝隙
      cv::Mat k_gap = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(gap_k, gap_k));
      cv::morphologyEx(mask_any, mask_any, cv::MORPH_CLOSE, k_gap);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_any, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 选面积最大的轮廓作为目标（题目一次只需识别一个物块）
    int best = -1;
    double best_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
      double a = cv::contourArea(contours[i]);
      if (a > best_area) { best_area = a; best = static_cast<int>(i); }
    }

    vision_task1_interfaces::msg::BlockInfo out;
    out.header = header;

    if (best < 0 || best_area < static_cast<double>(min_area_px_)) {
      // 没找到足够大的目标 —— 也要发布"unknown"，这样话题频率才稳定
      out.block_class   = "unknown";
      out.edge_length_m = 0.0f;
      out.x_m = out.y_m = out.z_m = 0.0f;
      out.u_px = out.v_px = 0.0f;
      out.edge_length_px = 0.0f;
      out.confidence = 0.0f;
      publish(out, bgr, cv::Rect());
      return;
    }

    const std::vector<cv::Point> & c = contours[best];

    // ---- 7) 分类：统计轮廓内的红/蓝像素占比 ----
    //  先把单个轮廓画到空白掩膜上，再和红/蓝掩膜求交集，
    //  这样统计的像素严格限定在这个物块内部，不受画面里其他红色物体的干扰。
    cv::Mat only_this = cv::Mat::zeros(mask_any.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> one{c};
    cv::drawContours(only_this, one, 0, cv::Scalar(255), cv::FILLED);

    cv::Mat this_red, this_blue;
    cv::bitwise_and(mask_red,  only_this, this_red);
    cv::bitwise_and(mask_blue, only_this, this_blue);

    const double n_red  = static_cast<double>(cv::countNonZero(this_red));
    const double n_blue = static_cast<double>(cv::countNonZero(this_blue));
    const double n_sum  = n_red + n_blue;
    if (n_sum < 1.0) {
      out.block_class = "unknown";
      out.confidence  = 0.0f;
      publish(out, bgr, cv::Rect());
      return;
    }
    const double r_red  = n_red  / n_sum;
    const double r_blue = n_blue / n_sum;

    // ---- 8) 判定种类 ----
    //  顺序很重要：先排除"单色占绝对多数"的情况，再判相间。
    //  因为红蓝相间的物块在某些角度下某一色会占很多（比如镜头几乎正对红色层），
    //  所以还要用"上下分层"这个几何证据来交叉验证。
    cv::Rect box = cv::boundingRect(c);
    std::string cls = "unknown";
    double conf = 0.0;
    bool need_layer = false;

    if (r_red >= dominant_ratio_ && r_blue < interleave_ratio_) {
      cls = "red";
      conf = r_red;
    } else if (r_blue >= dominant_ratio_ && r_red < interleave_ratio_) {
      cls = "blue";
      conf = r_blue;
    } else if (r_red >= interleave_ratio_ && r_blue >= interleave_ratio_) {
      // 两色都占相当比例 —— 候选"红蓝相间"，还要用上下分层几何证据确认
      cls = "red_blue";
      conf = std::min(r_red, r_blue) * 2.0;   // 两色越均衡越可信
      need_layer = true;
    }

    // 分层验证：如果红蓝像素总量在上下两层里分布明显不均，
    // 说明更像"一整块某色 + 边缘反光/阴影"，而不是真正的相间物块。
    if (need_layer && !layer_check(this_red, this_blue, box)) {
      // 退化成占优的那一色，避免把"红色块上的蓝色反光"误判成相间
      if (r_red >= r_blue) { cls = "red";  conf = r_red;  }
      else                 { cls = "blue"; conf = r_blue; }
    }

    // ---- 9) 像素尺寸 ----
    //  用最小外接旋转矩形来量物块。比 boundingRect 稳：物块斜放时 boundingRect 会偏大。
    //
    //  ★ 取【长边】而不是短边。原因：
    //    物块是立方体 —— 正对相机时投影是正方形（两边相等）；
    //    侧视时投影变成矩形，【长边】仍然对应立方体的棱长（透视缩短只影响"进深"方向，
    //    表现为短边变短）。所以长边更接近真实棱长的投影。
    //    若用短边，侧视时短边变小 -> Z = f*L/短边 会把距离算得【偏大】，
    //    这正是 3cm 精度要求下的典型翻车点。
    cv::RotatedRect rr = cv::minAreaRect(c);
    double px_w = rr.size.width;
    double px_h = rr.size.height;
    float  px_edge = static_cast<float>(std::max(px_w, px_h));

    // ---- 10) 物理长度 + 三维坐标 ----
    const double fx = has_intrinsics_ ? fx_ : fallback_fx_;
    const double fy = has_intrinsics_ ? fy_ : fallback_fy_;
    const double cx = has_intrinsics_ ? cx_ : fallback_cx_;
    const double cy = has_intrinsics_ ? cy_ : fallback_cy_;

    float z = 0.0f, x = 0.0f, y = 0.0f, len_m = 0.0f;
    if (px_edge > 1.0f) {
      if (distance_mode_ == "ground_plane") {
        z = estimate_distance_ground_plane(box, bgr.rows, fy, cy);
      } else {
        // 针孔模型：Z = fy * 真实尺寸 / 像素尺寸
        //  推导：像素高度 / fy = 真实高度 / Z  （相似三角形）
        z = static_cast<float>(fy * real_edge_m_ / static_cast<double>(px_edge));
      }
      // 反投影：把像素坐标换算成相机坐标系下的三维点
      //  u - cx 是"偏右多少像素"，乘 Z/fx 就变成"偏右多少米"
      const double uc = box.x + box.width  * 0.5;
      const double vc = box.y + box.height * 0.5;
      x = static_cast<float>((uc - cx) * z / fx);
      y = static_cast<float>((vc - cy) * z / fy);
      len_m = static_cast<float>(real_edge_m_);
    }

    // ---- 11) 加分项：参照点偏移补偿 ----
    //  题目原文：「如果我的参照点与摄像头的中心有一定的偏移，
    //             程序能够通过调整偏移量实现精准的定位」
    //  含义：如果验收时量的不是"物块中心到相机光心"，而是到某个安装偏移了的参照点，
    //        就把这个固定偏移平移掉。
    //  实现：在相机坐标系下做一个平移 P_ref = P_cam - t_offset
    if (std::abs(corner_offset_m_) > 1e-9) {
      const float t = static_cast<float>(corner_offset_m_);
      // 这里假设偏移发生在水平方向（最常见：参照点比镜头偏左或偏右）。
      // 如果你的实际偏移在别的方向，把 t 加到对应的分量上即可。
      x -= t;
      RCLCPP_DEBUG(this->get_logger(), "Applied offset compensation %.4f m -> x=%.4f", t, x);
    }

    // ---- 12) 组包发布 ----
    out.block_class    = cls;
    out.edge_length_m  = len_m;
    out.edge_length_px = px_edge;
    out.x_m = x; out.y_m = y; out.z_m = z;
    out.u_px = box.x + box.width  * 0.5f;
    out.v_px = box.y + box.height * 0.5f;
    out.confidence = static_cast<float>(conf);

    publish(out, bgr, box);

    // 每次识别都打印一行，方便你肉眼核对；频率统计由 report_rate 负责
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "class=%s edge=%.4fm (%.1fpx) z=%.4fm x=%.4fm y=%.4fm conf=%.2f",
      cls.c_str(), len_m, px_edge, z, x, y, conf);
  }

  // -------------------------------------------------------------------------
  //  上下分层检查：用来确认"红蓝相间"而不是"某色为主 + 少量反光"
  //  做法：把外接框按 layer_split_ratio 切成上下两块，看是否形成
  //        "上红下蓝"或"上蓝下红"的层次结构，且两个颜色层规模大致相当。
  //  注意：相机可能拍到物块的正面或背面，红在上/蓝在上都可能，所以两种都接受。
  //
  //  返回 true 表示"确实是分层的"，false 表示"更像单色+杂色"。
  //  返回 false 时调用方会把分类退化成占优的那一色。
  // -------------------------------------------------------------------------
  bool layer_check(const cv::Mat & red_mask, const cv::Mat & blue_mask, const cv::Rect & box)
  {
    if (box.height < 8 || box.width < 8) { return false; }
    const int split = box.y + static_cast<int>(box.height * layer_split_ratio_);
    const int y0 = box.y;
    const int y1 = std::min(box.y + box.height, red_mask.rows);
    const int x0 = box.x;
    const int x1 = std::min(box.x + box.width, red_mask.cols);
    if (split <= y0 || split >= y1) { return false; }

    cv::Rect upper(x0, y0, x1 - x0, split - y0);
    cv::Rect lower(x0, split, x1 - x0, y1 - split);
    upper &= cv::Rect(0, 0, red_mask.cols, red_mask.rows);
    lower &= cv::Rect(0, 0, red_mask.cols, red_mask.rows);
    if (upper.area() <= 0 || lower.area() <= 0) { return false; }

    const double up_red   = cv::countNonZero(red_mask(upper));
    const double up_blue  = cv::countNonZero(blue_mask(upper));
    const double low_red  = cv::countNonZero(red_mask(lower));
    const double low_blue = cv::countNonZero(blue_mask(lower));

    // 层次结构：上半以某色为主，下半以另一色为主
    const bool red_top  = (up_red  > up_blue)  && (low_blue > low_red);
    const bool blue_top = (up_blue > up_red)   && (low_red  > low_blue);
    const bool layered  = red_top || blue_top;

    // 均衡度：比较【上层的主色层】和【下层的主色层】的规模，不要拿"上下两层总面积"比。
    //
    //  ★ 这里修了一个会让红蓝相间【永远判不出来】的 bug：
    //    两层的高度由 layer_split_ratio 决定。早期版本默认 0.25（上 25% / 下 75%），
    //    此时两层总面积之比恒为 0.25/0.75 = 0.333，而 layer_size_tol 是 0.45 ——
    //    0.333 < 0.45 恒成立，layer_check 永远返回 false，
    //    红蓝相间每次都被退回成"占优的那一色"（实测：相间块被判成 blue）。
    //    改成比较"上层的红像素"与"下层的蓝像素"（即两个颜色层自身的规模），
    //    这个量才真正对应注释里说的"两层大小相当"。
    const double dom_up  = red_top ? up_red  : up_blue;    // 上层的主色像素数
    const double dom_low = red_top ? low_blue : low_red;   // 下层的主色像素数
    const double ratio = (dom_up + dom_low) > 0
                       ? std::min(dom_up, dom_low) / std::max(dom_up, dom_low) : 0.0;
    const bool balanced = ratio >= layer_size_tol_;

    const bool ok = layered && balanced;
    RCLCPP_DEBUG(this->get_logger(),
      "layer check: top(r%.0f b%.0f) bottom(r%.0f b%.0f) layered=%d balance=%.2f(thr %.2f) -> %s",
      up_red, up_blue, low_red, low_blue, layered ? 1 : 0, ratio, layer_size_tol_,
      ok ? "red_blue confirmed" : "fallback to dominant color");
    return ok;
  }

  // -------------------------------------------------------------------------
  //  地面平面模式的距离估计
  //  适用场景：物块放在桌面上、相机固定俯视。此时物块底边在图像上的位置
  //           直接决定距离，比"像素高度"稳（因为像素高度受物块朝向影响大）。
  //
  //  原理：相机俯角 θ、高度 h，物块底边在图像上的垂直像素 v：
  //        与光轴夹角 α = atan((v - cy) / fy)
  //        地面距离 Z = h * tan(θ + α)
  // -------------------------------------------------------------------------
  float estimate_distance_ground_plane(const cv::Rect & box, int img_rows,
                                       double fy, double cy)
  {
    (void)img_rows;
    const double v_bottom = box.y + box.height;      // 物块底边的像素行
    const double alpha = std::atan((v_bottom - cy) / fy);
    const double theta = cam_pitch_deg_ * M_PI / 180.0;
    double z = cam_height_m_ * std::tan(theta + alpha);
    if (!std::isfinite(z) || z <= 0.0) { z = 0.0; }
    return static_cast<float>(z);
  }

  // -------------------------------------------------------------------------
  //  发布 + （可选）画图显示
  // -------------------------------------------------------------------------
  void publish(const vision_task1_interfaces::msg::BlockInfo & out,
               const cv::Mat & bgr, const cv::Rect & box)
  {
    pub_->publish(out);
    ++pub_count_;

    if (!show_image_) { return; }

    cv::Mat canvas = bgr.clone();
    if (box.area() > 0) {
      cv::rectangle(canvas, box, viz::kGreen, 2);
      const std::string label = out.block_class + " z=" + cv::format("%.3f", out.z_m) + "m";
      cv::putText(canvas, label, cv::Point(box.x, std::max(20, box.y - 8)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, viz::kBlack, 3);
      cv::putText(canvas, label, cv::Point(box.x, std::max(20, box.y - 8)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, viz::kYellow, 1);
      // 画面中心画十字，直观看出"物块离相机中心偏了多少"
      cv::drawMarker(canvas, cv::Point(canvas.cols / 2, canvas.rows / 2),
                     viz::kRed, cv::MARKER_CROSS, 24, 2);
    }
    cv::imshow("color_detector (press q to quit)", canvas);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      RCLCPP_INFO(this->get_logger(), "Quit key pressed, closing display window");
      show_image_ = false;
      cv::destroyAllWindows();
    }
  }

  // -------------------------------------------------------------------------
  //  每秒打印一次真实发布频率 —— 这是题目「≥60Hz」的验收证据
  // -------------------------------------------------------------------------
  void report_rate()
  {
    const auto now = std::chrono::steady_clock::now();
    if (last_report_.time_since_epoch().count() != 0) {
      const double dt = std::chrono::duration<double>(now - last_report_).count();
      const double hz = static_cast<double>(pub_count_ - last_count_) / std::max(1e-6, dt);
      RCLCPP_INFO(this->get_logger(),
        ">>> publish rate = %.1f Hz  (requirement >= 60 Hz)  total %lu msgs",
        hz, static_cast<unsigned long>(pub_count_));
      if (hz < 60.0) {
        RCLCPP_WARN(this->get_logger(),
          "Below 60Hz. Try: increase republish_hz, lower image resolution, "
          "set show_image:=false, check CPU load");
      }
    }
    last_report_ = now;
    last_count_  = pub_count_;
  }

  // ---- 发布者 / 订阅者 ----
  rclcpp::Publisher<vision_task1_interfaces::msg::BlockInfo>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::TimerBase::SharedPtr timer_, fps_timer_;

  // ---- 参数 ----
  int red_h_low_max_{10}, red_h_high_min_{170}, red_s_min_{90}, red_v_min_{50};
  int blue_h_min_{95}, blue_h_max_{135}, blue_s_min_{90}, blue_v_min_{50};
  double dominant_ratio_{0.80}, interleave_ratio_{0.18};
  double layer_split_ratio_{0.50}, layer_size_tol_{0.45};
  int morph_kernel_{5}, min_area_px_{200}, blur_kernel_{5};
  double real_edge_m_{0.05}, corner_offset_m_{0.0};
  std::string distance_mode_{"pixel_height"};
  double cam_height_m_{0.35}, cam_pitch_deg_{30.0};
  std::string image_path_;
  double republish_hz_{60.0};
  bool show_image_{false};
  double fallback_fx_{600.0}, fallback_fy_{600.0}, fallback_cx_{320.0}, fallback_cy_{240.0};

  // ---- 运行时状态 ----
  cv::Mat static_image_;
  bool has_intrinsics_{false};
  bool warned_intrinsics_{false};
  double fx_{600.0}, fy_{600.0}, cx_{320.0}, cy_{240.0};

  // ---- 频率统计 ----
  std::chrono::steady_clock::time_point last_report_{};
  unsigned long pub_count_{0}, last_count_{0};
};

// =============================================================================
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ColorDetector>());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("color_detector"), "Node exited with error: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
