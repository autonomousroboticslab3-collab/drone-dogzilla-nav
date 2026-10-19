// yolo.h

#ifndef YOLO_H
#define YOLO_H

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>
#include <list>

using namespace std;
using namespace cv;
using namespace dnn;

namespace ORB_SLAM3 {
    class yolov3Segment {
        private:
            float confThreshold = 0.5; // Confidence threshold
            float nmsThreshold = 0.4;  // Non-maximum suppression threshold
            int inpWidth = 512;  // Width of network's input image
            int inpHeight = 512; // Height of network's input image
            vector<string> classes;
            Net net;

        public:
            pair<int, int> box_c;
            vector<int> boxx;
            vector<vector<int>> vvbox;
            pair<cv::Mat, vector<vector<int>> > result;
            int center_y;
            yolov3Segment();
            ~yolov3Segment();
            pair<cv::Mat, vector<vector<int>>> Segmentation(cv::Mat &image);
            
            // Remove the bounding boxes with low confidence using non-maxima suppression
            cv::Mat postprocess(Mat& frame, const vector<Mat>& out);
            // draw prediction
            void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
            // getcoordinate
            vector<int> getcoordinate(Mat& frame);

            // test
            std::pair<int, int> coordinate(cv::Mat &image);
            
            // Get the names of the output layers
            vector<String> getOutputsNames(const Net& net);
            //判斷是否在Box裡
            // bool IsInBox(const int& i, int& box_id);

    };
}

#endif // YOLO_H