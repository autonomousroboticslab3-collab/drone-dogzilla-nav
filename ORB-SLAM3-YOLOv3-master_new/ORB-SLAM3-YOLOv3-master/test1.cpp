/**
* This file is part of ORB-SLAM3
*
* [版權聲明省略]
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h> // for usleep
#include <opencv2/opencv.hpp>
#include "System.h"
#include "yolo.h" // 確保 yolo.h 定義了 ORB_SLAM3 命名空間和 yolov3Segment 類別
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <exception>
#include <vector>
#include <sys/stat.h> // for mkdir
#include <sys/types.h>
#include <dirent.h> // for directory traversal
#include <sstream>
#include <iomanip>
#include <errno.h>

// 定義 make_unique 以支援 C++11，並放在自訂命名空間中避免與 std::make_unique 衝突
namespace cxx11 {
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args){
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

using namespace std;

// 線程安全的有限佇列模板
template <typename T>
class SafeQueue {
private:
    queue<T> q;
    mutable mutex m;
    condition_variable cv;
    size_t max_size;
public:
    SafeQueue(size_t max_size_) : max_size(max_size_) {}

    void enqueue(T item) {
        unique_lock<mutex> lock(m);
        if (q.size() >= max_size) {
            q.pop(); // 丟棄最舊的幀
            cout << "[SafeQueue] Queue full, discarding oldest frame." << endl;
        }
        q.push(move(item));
        cv.notify_one();
    }

    bool dequeue(T& item) {
        unique_lock<mutex> lock(m);
        while (q.empty()) {
            if(cv.wait_for(lock, chrono::milliseconds(100)) == std::cv_status::timeout){
                return false;
            }
        }
        if (!q.empty()) {
            item = move(q.front());
            q.pop();
            return true;
        }
        return false;
    }

    void clear() {
        unique_lock<mutex> lock(m);
        while (!q.empty()) q.pop();
    }
};

// 函式來檢查目錄是否存在
bool directoryExists(const string& path) {
    struct stat info;
    if(stat(path.c_str(), &info) != 0)
        return false;
    return (info.st_mode & S_IFDIR);
}

// 函式來建立目錄
bool createDirectory(const string& path) {
    // 0755 權限
    if(mkdir(path.c_str(), 0755) == -1) {
        if(errno == EEXIST) {
            return true; // 目錄已存在
        } else {
            return false; // 其他錯誤
        }
    }
    return true;
}

// 函式來將 cv::Mat 轉換為 Eigen::Matrix4f
Eigen::Matrix4f cvMatToEigen(const cv::Mat& mat){
    Eigen::Matrix4f eigenMat;
    for(int i=0; i<4; ++i){
        for(int j=0; j<4; ++j){
            eigenMat(i,j) = mat.at<float>(i,j);
        }
    }
    return eigenMat;
}

int main(int argc, char **argv)
{
    // 硬編碼路徑設定
    // 請確保這些檔案在你的目錄下路徑正確
    // 找到這幾行，確保 parameterFile 指向你的目錄
    string vocFile = "/home/t1204/Downloads/ORB-SLAM3-YOLOv3-master/Vocabulary/ORBvoc.txt";
    // 修正這行：
    string parameterFile = "/home/t1204/Downloads/ORB-SLAM3-YOLOv3-master/Examples/Monocular/dji.yaml";
    string outputDir = "/home/t1204/Downloads/ORB-SLAM3-YOLOv3-master/Examples/Monocular/";
    // 初始化 YOLO
    unique_ptr<ORB_SLAM3::yolov3Segment> yolo;
    try {
        cout << "[Main] Loading Yolov3 net. This could take a while..." << endl;
        yolo = cxx11::make_unique<ORB_SLAM3::yolov3Segment>();
        cout << "[Main] Yolov3 net loaded!" << endl;
    } catch (const exception& e) {
        cerr << "[Main] Exception while initializing YOLO: " << e.what() << endl;
        return -1;
    }

    // 初始化 ORB-SLAM3
    unique_ptr<ORB_SLAM3::System> SLAM;
    try {
        cout << "[Main] Initializing ORB-SLAM3..." << endl;
        SLAM = cxx11::make_unique<ORB_SLAM3::System>(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);
        cout << "[Main] ORB-SLAM3 initialized." << endl;
    } catch (const exception& e) {
        cerr << "[Main] Exception while initializing ORB-SLAM3: " << e.what() << endl;
        return -1;
    }

    // 初始化相機
    cout << "[Main] Opening video stream..." << endl;
    // cv::VideoCapture cap("rtmp://100.87.245.5:1935/live1/stream"); // 使用 RTMP 流
    // 如果是 USB 相機，可以改為 0
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[Main] Error: Could not open the camera!" << endl;
        return -1;
    }
    cout << "[Main] Video stream opened successfully." << endl;

    // 創建 KITTI 格式儲存目錄
    string kittiImagesDir = outputDir + "kitti_images";
    string kittiPosesFile = outputDir + "poses.txt";
    if(!directoryExists(kittiImagesDir)) {
        if(!createDirectory(kittiImagesDir)) {
            cerr << "[Main] Error: Could not create KITTI images directory at " << kittiImagesDir << "!" << endl;
            return -1;
        }
        cout << "[Main] KITTI images directory created at " << kittiImagesDir << "." << endl;
    } else {
        cout << "[Main] KITTI images directory already exists at " << kittiImagesDir << "." << endl;
    }

    // 打開文件來儲存 KITTI 位姿數據
    ofstream kittiPosesOut;
    kittiPosesOut.open(kittiPosesFile);
    if (!kittiPosesOut.is_open()) {
        cerr << "[Main] Error: Could not open KITTI poses file at " << kittiPosesFile << "!" << endl;
        return -1;
    }
    cout << "[Main] KITTI poses file opened at " << kittiPosesFile << "." << endl;

    // 打開文件來儲存 TUM 格式的位姿數據
    ofstream tumFile;
    string tumFilePath = outputDir + "slam_poses.txt";
    tumFile.open(tumFilePath);
    if (!tumFile.is_open()) {
        cerr << "[Main] Error: Could not open TUM file at " << tumFilePath << "!" << endl;
        return -1;
    }
    cout << "[Main] TUM file opened at " << tumFilePath << "." << endl;

    // 定義佇列，設置最大長度以防止積壓
    const size_t FRAME_QUEUE_MAX_SIZE = 3; // 減少佇列大小以防止積壓
    SafeQueue<shared_ptr<cv::Mat>> slamQueue(FRAME_QUEUE_MAX_SIZE);
    SafeQueue<shared_ptr<cv::Mat>> yoloQueue(FRAME_QUEUE_MAX_SIZE);
    SafeQueue<pair<shared_ptr<cv::Mat>, double>> displayQueue(FRAME_QUEUE_MAX_SIZE); // 用於顯示 SLAM 結果
    SafeQueue<pair<shared_ptr<cv::Mat>, vector<vector<int>>>> yoloDisplayQueue(FRAME_QUEUE_MAX_SIZE); // 用於顯示 YOLO 檢測結果

    // 追蹤時間統計相關
    vector<float> vTimesTrack;
    mutex timesMutex;

    // 記錄開始時間
    auto start = chrono::steady_clock::now();

    // 控制程式退出的旗標
    atomic<bool> isRunning(true);

    // 捕捉影像的執行緒
    thread captureThread([&cap, &slamQueue, &yoloQueue, &isRunning]() {
        cout << "[CaptureThread] Started." << endl;
        try {
            while (isRunning.load()) {
                shared_ptr<cv::Mat> frame = make_shared<cv::Mat>();
                if (!cap.read(*frame)) {
                    cerr << "[CaptureThread] Error: Could not read frame from camera." << endl;
                    isRunning.store(false);
                    break;
                }
                // 檢查影像是否有效
                if (frame->empty()) {
                    cerr << "[CaptureThread] Warning: Captured empty frame." << endl;
                    continue;
                }
                slamQueue.enqueue(frame);
                yoloQueue.enqueue(frame);
                // 控制幀率，例如每幀間隔10毫秒（約100幀/秒）
                this_thread::sleep_for(chrono::milliseconds(10));
            }
        } catch (const exception& e) {
            cerr << "[CaptureThread] Exception: " << e.what() << endl;
            isRunning.store(false);
        }
        cout << "[CaptureThread] Exiting." << endl;
    });

    // SLAM 處理的執行緒
    thread slamThread([&SLAM, &slamQueue, &displayQueue, &start, &isRunning, &tumFile, &vTimesTrack, &timesMutex, &kittiImagesDir, &kittiPosesOut]() {
        cout << "[SLAMThread] Started." << endl;
        int frameCount = 0;
        try {
            while (isRunning.load()) {
                shared_ptr<cv::Mat> frame;
                if (slamQueue.dequeue(frame)) {
                    // 縮放影像以加快處理速度，根據需要調整
                    cv::Mat frame_resized;
                    cv::resize(*frame, frame_resized, cv::Size(960, 540));

                    if (frame_resized.empty()) {
                        cerr << "[SLAMThread] Error: Resized frame is empty." << endl;
                        continue; // 跳過此幀
                    }

                    // 計算時間戳
                    auto now = chrono::steady_clock::now();
                    auto duration = chrono::duration_cast<chrono::milliseconds>(now - start).count();
                    double timestamp = duration / 1000.0;

                    // 開始計時
                    auto trackingStart = chrono::steady_clock::now();

                    // SLAM 處理
                    cout << "[SLAMThread] Tracking frame " << frameCount << " with timestamp: " << timestamp << endl;
                    cv::Mat Tcw = SLAM->TrackMonocular(frame_resized, timestamp); // 獲取 SLAM 系統的位姿矩陣 (4x4)
                    cout << "[SLAMThread] Frame " << frameCount << " tracked." << endl;

                    // 結束計時
                    auto trackingEnd = chrono::steady_clock::now();
                    chrono::duration<float, std::milli> trackingDuration = trackingEnd - trackingStart;
                    float trackingTime = trackingDuration.count(); // 單位：毫秒

                    // 儲存追蹤時間
                    {
                        lock_guard<mutex> lock(timesMutex);
                        vTimesTrack.push_back(trackingTime);
                    }

                    if (!Tcw.empty()) {
                        // 確保 Tcw 為 4x4
                        if (Tcw.rows != 4 || Tcw.cols != 4) {
                            cerr << "[SLAMThread] Error: Tcw is not a 4x4 matrix." << endl;
                            continue;
                        }

                        // 確保 Tcw 是 float 類型
                        if (Tcw.type() != CV_32F) {
                            Tcw.convertTo(Tcw, CV_32F);
                        }

                        // 將 Tcw 轉換為 Eigen::Matrix4f
                        Eigen::Matrix4f Tcw_eigen = cvMatToEigen(Tcw);

                        // 計算 Twc = inverse(Tcw)
                        Eigen::Matrix4f Twc = Tcw_eigen.inverse();

                        // 提取平移向量 (tx, ty, tz)
                        Eigen::Vector3f translation(Twc(0, 3), Twc(1, 3), Twc(2, 3));

                        // 提取旋轉矩陣並轉換為四元數
                        Eigen::Matrix3f rotation = Twc.block<3,3>(0,0);
                        Eigen::Quaternionf quaternion(rotation);

                        // 將位姿數據寫入 TUM 文件
                        tumFile << fixed << setprecision(6)
                                << timestamp << " "
                                << translation.x() << " " << translation.y() << " " << translation.z() << " "
                                << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w() << endl;

                        // 將位姿數據寫入 KITTI poses 文件（行格式：R (3x4) t (3x1)）
                        // KITTI 位姿格式：行優先的 3x4 矩陣
                        // 我們將使用 Twc 來表示從世界座標系到相機座標系的轉換

                        stringstream ss;
                        for(int i=0; i<3; ++i){
                            for(int j=0; j<4; ++j){
                                ss << fixed << setprecision(6) << Twc(i,j) << " ";
                            }
                        }
                        ss << "\n";
                        kittiPosesOut << ss.str();

                        // 儲存影像為 KITTI 格式
                        // 例如：000000.png, 000001.png, ...
                        stringstream imgName;
                        imgName << kittiImagesDir << "/" << setfill('0') << setw(6) << frameCount << ".png";
                        cv::imwrite(imgName.str(), frame_resized);
                    }

                    // 將處理後的影像和時間戳放入 displayQueue
                    displayQueue.enqueue(make_pair(make_shared<cv::Mat>(frame_resized), timestamp));

                    frameCount++;
                }
            }
        } catch (const exception& e) {
            cerr << "[SLAMThread] Exception: " << e.what() << endl;
            isRunning.store(false);
        }
        cout << "[SLAMThread] Exiting." << endl;
    });

    // YOLO 處理的執行緒
    thread yoloThread([&yolo, &yoloQueue, &yoloDisplayQueue, &isRunning]() {
        cout << "[YOLOThread] Started." << endl;
        try {
            while (isRunning.load()) {
                shared_ptr<cv::Mat> frame;
                if (yoloQueue.dequeue(frame)) {
                    // 縮放影像以適應 YOLO 網路的輸入尺寸
                    cv::Mat frame_resized;
                    cv::resize(*frame, frame_resized, cv::Size(512, 512)); // 512x512 用於 YOLO

                    if (frame_resized.empty()) {
                        cerr << "[YOLOThread] Error: Resized frame is empty." << endl;
                        continue; // 跳過此幀
                    }

                    // YOLO 進行物件檢測並獲得檢測結果
                    cout << "[YOLOThread] Running YOLO segmentation..." << endl;
                    auto segmentationResult = yolo->Segmentation(frame_resized);
                    cout << "[YOLOThread] YOLO segmentation completed." << endl;

                    // 檢查 Segmentation 結果是否有效
                    if (segmentationResult.first.empty()) {
                        cout << "[YOLOThread] Segmentation failed, skipping frame." << endl;
                        continue; // 跳過此幀
                    }

                    // 獲取檢測結果
                    cv::Mat yolo_mask = segmentationResult.first;
                    vector<vector<int>> detections = segmentationResult.second;

                    // 繪製 BoundingBox
                    for (const auto &box : detections) {
                        if (box.size() < 5) continue; // 確保 box 包含足夠的信息
                        int classId = box[0];
                        int left = box[1], top = box[2], right = box[3], bottom = box[4];
                        yolo->drawPred(classId, 1.0, left, top, right, bottom, frame_resized); // 繪製框
                    }

                    // 將 YOLO 檢測結果與影像放入 yoloDisplayQueue
                    yoloDisplayQueue.enqueue(make_pair(make_shared<cv::Mat>(frame_resized), detections));
                }
            }
        } catch (const exception& e) {
            cerr << "[YOLOThread] Exception: " << e.what() << endl;
            isRunning.store(false);
        }
        cout << "[YOLOThread] Exiting." << endl;
    });

    // 顯示影像的執行緒
    thread displayThread([&displayQueue, &yoloDisplayQueue, &isRunning]() {
        cout << "[DisplayThread] Started." << endl;
        try {
            while (isRunning.load()) {
                // 顯示 SLAM 結果
                pair<shared_ptr<cv::Mat>, double> slamData;
                if (displayQueue.dequeue(slamData)) {
                    cv::imshow("SLAM Frame", *(slamData.first));
                }

                // 顯示 YOLO 檢測結果
                pair<shared_ptr<cv::Mat>, vector<vector<int>>> yoloData;
                if (yoloDisplayQueue.dequeue(yoloData)) {
                    cv::imshow("YOLO Detections", *(yoloData.first));
                }

                // 檢查是否按下 'q' 鍵來退出
                char key = (char)cv::waitKey(1);
                if (key == 'q' || key == 'Q') {
                    cout << "[DisplayThread] Exit key pressed." << endl;
                    isRunning.store(false);
                    break;
                }
            }
        } catch (const exception& e) {
            cerr << "[DisplayThread] Exception: " << e.what() << endl;
            isRunning.store(false);
        }
        cout << "[DisplayThread] Exiting." << endl;
    });

    // 等待所有線程結束
    if (captureThread.joinable()) captureThread.join();
    if (slamThread.joinable()) slamThread.join();
    if (yoloThread.joinable()) yoloThread.join();
    if (displayThread.joinable()) displayThread.join();

    // Shutdown SLAM
    cout << "[Main] Shutting down SLAM..." << endl;
    SLAM->Shutdown(); // 關閉 SLAM 系統
    cout << "[Main] SLAM shut down." << endl;

    // 關閉 TUM 文件
    tumFile.close();
    cout << "[Main] TUM file closed." << endl;

    // 關閉 KITTI poses 文件
    kittiPosesOut.close();
    cout << "[Main] KITTI poses file closed." << endl;

    // 計算追蹤時間統計
    vector<float> sortedTimes;
    {
        lock_guard<mutex> lock(timesMutex);
        sortedTimes = vTimesTrack;
    }
    if(!sortedTimes.empty()){
        sort(sortedTimes.begin(), sortedTimes.end());
        float totaltime = 0.0f;
        for(int ni = 0; ni < sortedTimes.size(); ni++)
        {
            totaltime += sortedTimes[ni];
        }

        float medianTime;
        if(sortedTimes.size() % 2 == 0){
            medianTime = (sortedTimes[sortedTimes.size()/2 - 1] + sortedTimes[sortedTimes.size()/2]) / 2.0f;
        }
        else{
            medianTime = sortedTimes[sortedTimes.size()/2];
        }

        float meanTime = totaltime / sortedTimes.size();

        cout << "-------" << endl << endl;
        cout << "Median tracking time: " << medianTime << " ms" << endl;
        cout << "Mean tracking time: " << meanTime << " ms" << endl;

        // 將統計數據寫入txt檔
        ofstream outFile("tracking_time_stats.txt");
        if (!outFile) {
            cerr << "[Main] Failed to create output file for tracking time statistics." << endl;
        }
        else {
            outFile << "Tracking Time Statistics\n";
            outFile << "-------------------------\n";
            outFile << "Number of Frames: " << sortedTimes.size() << "\n\n";

            outFile << "All Tracking Times (ms):\n";
            for(int ni = 0; ni < sortedTimes.size(); ni++)
            {
                outFile << sortedTimes[ni] << "\n";
            }

            outFile << "\nMedian Tracking Time: " << medianTime << " ms\n";
            outFile << "Mean Tracking Time: " << meanTime << " ms\n";

            outFile.close();

            cout << "[Main] Tracking time statistics saved to tracking_time_stats.txt" << endl;
        }
    }
    else{
        cout << "[Main] No tracking times recorded." << endl;
    }

    // 釋放 YOLO 物件
    yolo.reset();
    cout << "[Main] YOLO object released." << endl;

    return 0;
}