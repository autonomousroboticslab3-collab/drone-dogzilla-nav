/**
 * This file is part of ORB-SLAM3
 *
 * [版權聲明省略]
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h> // for usleep
#include <opencv2/opencv.hpp>
#include "System.h"
#include "yolo.h" // 確保 yolo.h 定義了 ORB_SLAM3 命名空間和 yolov3Segment 類別
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

int main(int argc, char **argv)
{
    // 檢查參數數量
    if(argc != 3 && argc != 4)
    {
        cerr << endl << "Usage: ./mono_camera path_to_vocabulary path_to_settings [path_to_masks]" << endl;
        return 1;
    }

    // 硬編碼路徑設定（根據實際情況修改）
    string vocFile = argv[1]; // 詞彙表路徑
    string parameterFile = argv[2]; // 設定檔路徑
    string maskPath = "";
    if(argc == 4)
    {
        maskPath = argv[3]; // Mask 路徑
    }

    // 初始化 YOLO
    ORB_SLAM3::yolov3Segment* yolo = nullptr;
    if (argc == 4)
    {
        cout << "Loading Yolov3 net. This could take a while..." << endl;
        try
        {
            yolo = new ORB_SLAM3::yolov3Segment();
            cout << "Yolov3 net loaded!" << endl;
        }
        catch (const std::exception& e)
        {
            cerr << "Exception while initializing YOLO: " << e.what() << endl;
            return -1;
        }
    }

    // 創建 SLAM 系統。它初始化所有系統線程並準備處理幀。
    ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);
    cout << "[Main] ORB-SLAM3 initialized." << endl;

    // 打開相機（使用預設相機，通常是索引 0）
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cerr << "[Main] Error: Could not open the camera!" << endl;
        return -1;
    }
    cout << "[Main] Camera opened successfully." << endl;

    // 打開文件來儲存 TUM 格式的位姿數據
    string outputDir = "./"; // 可以根據需要修改輸出目錄
    ofstream tumFile;
    string tumFilePath = outputDir + "slam_poses.txt";
    tumFile.open(tumFilePath);
    if (!tumFile.is_open()) {
        cerr << "[Main] Error: Could not open TUM file at " << tumFilePath << "!" << endl;
        return -1;
    }
    cout << "[Main] TUM file opened at " << tumFilePath << "." << endl;

    // 主迴圈
    cv::Mat im;

    // 膨脹設置（如果需要，可以使用）
    int dilation_size = 15;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                              cv::Point(dilation_size, dilation_size));

    // 記錄開始時間
    auto start = chrono::steady_clock::now();

    cout << endl << "-------" << endl;
    cout << "Start processing camera stream ..." << endl << endl;

    while(true)
    {
        // 捕捉影像
        if(!cap.read(im))
        {
            cerr << "[Main] Error: Could not read frame from camera." << endl;
            break;
        }

        // 獲取時間戳（以秒為單位）
        auto now = chrono::steady_clock::now();
        chrono::duration<double> elapsed = now - start;
        double tframe = elapsed.count();

        if(im.empty())
        {
            cerr << "[Main] Warning: Captured empty frame." << endl;
            continue;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // 分割圖像
        int h = im.rows;
        int w = im.cols;
        cv::Mat mask = cv::Mat::ones(h, w, CV_8U);

        // 初始化檢測結果
        std::vector<std::vector<int>> detections;

        // 如果提供了 mask 路徑，則使用 YOLO 進行分割
        if (argc == 4 && yolo != nullptr) {
            auto segmentationResult = yolo->Segmentation(im);
            mask = segmentationResult.first; // 更新遮罩
            detections = segmentationResult.second; // 獲取檢測結果
        }

        // 將影像傳遞給 SLAM 系統
        SLAM.TrackMonocularyolo(im, mask, tframe, detections); // 確保傳遞的是 vector<vector<int>> 類型

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        // 等待以維持實時性（根據需要調整）
        // 例如，假設相機是 30 FPS，每幀間隔約 33 毫秒
        double frame_time = 1.0 / 30.0; // 30 FPS
        if(ttrack < frame_time)
            usleep((frame_time - ttrack) * 1e6); // usleep 需要微秒
    }

    // 停止所有線程
    SLAM.Shutdown();

    // 保存相機軌跡
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt"); // 添加
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // 釋放 YOLO 物件
    if(yolo != nullptr)
        delete yolo;

    // 關閉 TUM 文件
    tumFile.close();
    cout << "[Main] TUM file closed." << endl;

    return 0;
}