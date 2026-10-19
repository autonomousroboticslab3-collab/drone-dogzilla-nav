#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <time.h>
using namespace std;
using namespace cv;

namespace ORB_SLAM3{
    class yolov5{
        public:
            std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5)
            yolov5();
            ~yolov5();
            void yolov5::detect()
    };
}