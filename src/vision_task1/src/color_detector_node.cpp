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

namespace viz {
const cv::Scalar kRed    (  0,   0, 255);
const cv::Scalar kBlue   (255,   0,   0);
const cv::Scalar kGreen  (  0, 255,   0);
const cv::Scalar kYellow (  0, 255, 255);
const cv::Scalar kWhite  (255, 255, 255);
const cv::Scalar kBlack  (  0,   0,   0);
}

class ColorDetector : public rclcpp::Node
{
public:
  ColorDetector()
  : Node("color_detector")
  {

    red_h_low_max_   = declare_parameter("red_h_low_max",   10);
    red_h_high_min_  = declare_parameter("red_h_high_min", 170);
    red_s_min_       = declare_parameter("red_s_min",       90);
    red_v_min_       = declare_parameter("red_v_min",       50);

    blue_h_min_      = declare_parameter("blue_h_min",      95);
    blue_h_max_      = declare_parameter("blue_h_max",     135);
    blue_s_min_      = declare_parameter("blue_s_min",      90);
    blue_v_min_      = declare_parameter("blue_v_min",      50);

    dominant_ratio_     = declare_parameter("dominant_ratio",     0.80);
    interleave_ratio_   = declare_parameter("interleave_ratio",   0.18);
    layer_split_ratio_  = declare_parameter("layer_split_ratio",  0.50);
    layer_size_tol_     = declare_parameter("layer_size_tol",     0.45);

    morph_kernel_    = declare_parameter("morph_kernel",   5);
    min_area_px_     = declare_parameter("min_area_px",    200);
    blur_kernel_     = declare_parameter("blur_kernel",    5);

    real_edge_m_     = declare_parameter("real_edge_m",    0.05);
    corner_offset_m_ = declare_parameter("corner_offset_m", 0.0);
    distance_mode_   = declare_parameter("distance_mode",  std::string("pixel_height"));

    cam_height_m_    = declare_parameter("cam_height_m",   0.35);
    cam_pitch_deg_   = declare_parameter("cam_pitch_deg",  30.0);

    image_path_    = declare_parameter("image_path",   std::string(""));
    republish_hz_  = declare_parameter("republish_hz", 60.0);
    show_image_    = declare_parameter("show_image",   false);

    fallback_fx_ = declare_parameter("fallback_fx", 600.0);
    fallback_fy_ = declare_parameter("fallback_fy", 600.0);
    fallback_cx_ = declare_parameter("fallback_cx", 320.0);
    fallback_cy_ = declare_parameter("fallback_cy", 240.0);

    pub_ = this->create_publisher<vision_task1_interfaces::msg::BlockInfo>("block_info", 10);

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

    fps_timer_ = this->create_wall_timer(1s, std::bind(&ColorDetector::report_rate, this));

    RCLCPP_INFO(this->get_logger(),
      "param real_edge_m=%.4f m  <-- MEASURE YOUR BLOCK AND FIX THIS", real_edge_m_);
  }

private:

  void on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {

    if (msg->k.size() >= 6) {
      fx_ = msg->k[0]; fy_ = msg->k[4];
      cx_ = msg->k[2]; cy_ = msg->k[5];
      has_intrinsics_ = true;
    }
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (!has_intrinsics_) {

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

  void detect_and_publish(const cv::Mat & bgr, const std_msgs::msg::Header & header)
  {

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat hsv_blur;
    int bk = std::max(1, blur_kernel_ | 1);
    cv::GaussianBlur(hsv, hsv_blur, cv::Size(bk, bk), 0);

    cv::Mat mask_red_low, mask_red_high, mask_red;
    cv::inRange(hsv_blur,
      cv::Scalar(0, red_s_min_, red_v_min_),
      cv::Scalar(red_h_low_max_, 255, 255), mask_red_low);
    cv::inRange(hsv_blur,
      cv::Scalar(red_h_high_min_, red_s_min_, red_v_min_),
      cv::Scalar(179, 255, 255), mask_red_high);
    cv::bitwise_or(mask_red_low, mask_red_high, mask_red);

    cv::Mat mask_blue;
    cv::inRange(hsv_blur,
      cv::Scalar(blue_h_min_, blue_s_min_, blue_v_min_),
      cv::Scalar(blue_h_max_, 255, 255), mask_blue);

    int mk = std::max(1, morph_kernel_ | 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mk, mk));
    cv::morphologyEx(mask_red,  mask_red,  cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask_red,  mask_red,  cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask_blue, mask_blue, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask_blue, mask_blue, cv::MORPH_CLOSE, kernel);

    cv::Mat mask_any;
    cv::bitwise_or(mask_red, mask_blue, mask_any);

    {
      const int gap_k = std::max(7, mk * 2 + 1);
      cv::Mat k_gap = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(gap_k, gap_k));
      cv::morphologyEx(mask_any, mask_any, cv::MORPH_CLOSE, k_gap);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_any, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int best = -1;
    double best_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
      double a = cv::contourArea(contours[i]);
      if (a > best_area) { best_area = a; best = static_cast<int>(i); }
    }

    vision_task1_interfaces::msg::BlockInfo out;
    out.header = header;

    if (best < 0 || best_area < static_cast<double>(min_area_px_)) {

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

      cls = "red_blue";
      conf = std::min(r_red, r_blue) * 2.0;
      need_layer = true;
    }

    if (need_layer && !layer_check(this_red, this_blue, box)) {

      if (r_red >= r_blue) { cls = "red";  conf = r_red;  }
      else                 { cls = "blue"; conf = r_blue; }
    }

    cv::RotatedRect rr = cv::minAreaRect(c);
    double px_w = rr.size.width;
    double px_h = rr.size.height;
    float  px_edge = static_cast<float>(std::max(px_w, px_h));

    const double fx = has_intrinsics_ ? fx_ : fallback_fx_;
    const double fy = has_intrinsics_ ? fy_ : fallback_fy_;
    const double cx = has_intrinsics_ ? cx_ : fallback_cx_;
    const double cy = has_intrinsics_ ? cy_ : fallback_cy_;

    float z = 0.0f, x = 0.0f, y = 0.0f, len_m = 0.0f;
    if (px_edge > 1.0f) {
      if (distance_mode_ == "ground_plane") {
        z = estimate_distance_ground_plane(box, bgr.rows, fy, cy);
      } else {

        z = static_cast<float>(fy * real_edge_m_ / static_cast<double>(px_edge));
      }

      const double uc = box.x + box.width  * 0.5;
      const double vc = box.y + box.height * 0.5;
      x = static_cast<float>((uc - cx) * z / fx);
      y = static_cast<float>((vc - cy) * z / fy);
      len_m = static_cast<float>(real_edge_m_);
    }

    if (std::abs(corner_offset_m_) > 1e-9) {
      const float t = static_cast<float>(corner_offset_m_);

      x -= t;
      RCLCPP_DEBUG(this->get_logger(), "Applied offset compensation %.4f m -> x=%.4f", t, x);
    }

    out.block_class    = cls;
    out.edge_length_m  = len_m;
    out.edge_length_px = px_edge;
    out.x_m = x; out.y_m = y; out.z_m = z;
    out.u_px = box.x + box.width  * 0.5f;
    out.v_px = box.y + box.height * 0.5f;
    out.confidence = static_cast<float>(conf);

    publish(out, bgr, box);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "class=%s edge=%.4fm (%.1fpx) z=%.4fm x=%.4fm y=%.4fm conf=%.2f",
      cls.c_str(), len_m, px_edge, z, x, y, conf);
  }

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

    const bool red_top  = (up_red  > up_blue)  && (low_blue > low_red);
    const bool blue_top = (up_blue > up_red)   && (low_red  > low_blue);
    const bool layered  = red_top || blue_top;

    const double dom_up  = red_top ? up_red  : up_blue;
    const double dom_low = red_top ? low_blue : low_red;
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

  float estimate_distance_ground_plane(const cv::Rect & box, int img_rows,
                                       double fy, double cy)
  {
    (void)img_rows;
    const double v_bottom = box.y + box.height;
    const double alpha = std::atan((v_bottom - cy) / fy);
    const double theta = cam_pitch_deg_ * M_PI / 180.0;
    double z = cam_height_m_ * std::tan(theta + alpha);
    if (!std::isfinite(z) || z <= 0.0) { z = 0.0; }
    return static_cast<float>(z);
  }

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

  rclcpp::Publisher<vision_task1_interfaces::msg::BlockInfo>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::TimerBase::SharedPtr timer_, fps_timer_;

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

  cv::Mat static_image_;
  bool has_intrinsics_{false};
  bool warned_intrinsics_{false};
  double fx_{600.0}, fy_{600.0}, cx_{320.0}, cy_{240.0};

  std::chrono::steady_clock::time_point last_report_{};
  unsigned long pub_count_{0}, last_count_{0};
};

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
