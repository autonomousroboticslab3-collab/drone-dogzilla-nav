#ifndef YOLOV5_H
#define YOLOV5_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>

namespace ORB_SLAM3 {

    class yolov5 {
    public:
        // 建構函數：初始化 YOLOv5 模型和類別名稱
        yolov5(const std::string& model_path, const std::string& names_path);

        // 非極大值抑制 (Non-Max Suppression)
        std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

        // 執行偵測並在影像上繪製結果
        void detect(cv::Mat& frame, cv::Mat& output_frame);

    private:
        torch::jit::script::Module module;
        std::vector<std::string> classnames;
        clock_t start_time;
    };

}

#endif // YOLOV5_H