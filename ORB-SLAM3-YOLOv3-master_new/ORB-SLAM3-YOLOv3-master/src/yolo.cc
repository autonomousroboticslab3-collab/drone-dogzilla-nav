// yolo.cc

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include "yolo.h"
#include <vector>
#include <list>
#include <opencv2/dnn/shape_utils.hpp> //add
#include "Frame.h"

using namespace cv;
using namespace dnn;
using namespace std;

namespace ORB_SLAM3 {

yolov3Segment::yolov3Segment() {
    // Load names of classes
    string classesFile = "/home/t1204/Downloads/ORB-SLAM3-YOLOv3-master/src/yolo/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "/home/t1204/Downloads/ORB-SLAM3-YOLOv3-master/src/yolo/yolov3.cfg"; // 確保 cfg 和 weights 路徑正確
    String modelWeights = "/home/t1204/Downloads/ORB-SLAM3-YOLOv3-master/src/yolo/yolov3.weights";
    // String yolov5weight = "/home/t1102/Downloads/ORB-SLAM3-YOLOv3-master/src/yolo/yolov5s.onnx"; // 如需使用 YOLOv5

    // Load the network
    net = readNetFromDarknet(modelConfiguration, modelWeights);
    // net = readNetFromONNX(yolov5weight); // 如需使用 YOLOv5
    // net = readNetFromTensorflow(modelWeights); // 如需使用 TensorFlow

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 設置輸入尺寸
    inpWidth = 512;
    inpHeight = 512;

    // 設置閾值
    confThreshold = 0.5; // 可以根據需要調整
    nmsThreshold = 0.4;  // 可以根據需要調整
}

// Destructor definition
yolov3Segment::~yolov3Segment() {
    // 清理資源（如有需要）
}

// drawPred
void yolov3Segment::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
    // Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 0), 3);

    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!this->classes.empty()) {
        CV_Assert(classId < (int)this->classes.size());
        label = this->classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 2);
}

// detect
pair<cv::Mat, vector<vector<int>>> yolov3Segment::Segmentation(cv::Mat &image) {
    if (image.empty()) {
        cout << "error in yolo.cc: input image is empty." << endl;
        return make_pair(cv::Mat(), vector<vector<int>>());
    }

    cv::Mat blob;
    // Create a 4D blob from a frame.
    blobFromImage(image, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    // Sets the input to the network
    this->net.setInput(blob);
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    this->net.forward(outs, this->getOutputsNames(this->net)); // 確保 getOutputsNames 函數正確

    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    // Set mask size to match image
    int h = image.rows;
    int w = image.cols;
    cv::Mat mask = cv::Mat::ones(h,w,CV_8U);

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    vector<vector<int>> vvbox;
    pair<cv::Mat, vector<vector<int>>> result;

    // 解析 YOLO 的輸出
    for (size_t i = 0; i < outs.size(); ++i)
    {
        float* data = (float*)outs[i].data;

        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > this->confThreshold)
            {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                int right = centerX + width / 2;
                int bottom = centerY + height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        string c = this->classes[classIds[idx]];
        if (c == "person" || c == "car" || c == "bicycle" || c == "fire hydrant" ||
            c == "bottle" || c == "chair" || c == "tvmonitor" || c == "laptop" || c == "keyboard")
        {
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                box.x + box.width, box.y + box.height, image);

            // Initialize boxx correctly
            vector<int> boxx = {classIds[idx], box.x, box.y, box.x + box.width, box.y + box.height};
            vvbox.push_back(boxx);

            // Calculate center point
            int center_x = box.x + box.width / 2;
            int center_y = box.y + box.height / 2;
            cv::Point2i center(center_x, center_y);
            cv::circle(mask, center,10, Scalar(0,0,255), 10);
        }
    }

    // If mask size does not match image, report error and return empty
    if (mask.size() != image.size())
    {
        cout << "error in yolo.cc: mask size (" << mask.cols << "," << mask.rows << ") != image size (" << image.cols << "," << image.rows << ")" << endl;
        return make_pair(cv::Mat(), vector<vector<int>>());
    }

    // Dilate the mask
    cv::Mat mask_dilated;
    cv::dilate(mask, mask_dilated, kernel);
    mask = mask - mask_dilated;

    result.first = mask;
    result.second = vvbox;

    return result;
}

// Get the names of the output layers
vector<String> yolov3Segment::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // Get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

} // namespace ORB_SLAM3