#include <opencv2/opencv.hpp>
#include "System.h"
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>  // 用於文件寫入
#include "yolo.h"  // YOLO相關的頭文件
#include <Eigen/Core>  // 用於矩陣處理
#include <Eigen/Geometry>  // 用於四元數處理

using namespace std;

string parameterFile = "/home/t1102/Downloads/ORB-SLAM3-YOLOv3-master/Examples/Monocular/dji.yaml";  // 使用絕對路徑
string vocFile = "/home/t1102/Downloads/ORB-SLAM3-YOLOv3-master/Vocabulary/ORBvoc.txt";  // 使用絕對路徑
string datasetPath = "/home/t1102/Downloads/ORB-SLAM3-YOLOv3-master/Examples/Monocular/sequence";  // 評估資料集儲存路徑

int main(int argc, char **argv) {
    try {
        // 初始化 SLAM 系統
        cout << "Initializing SLAM system..." << endl;
        ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);
        cout << "SLAM system initialized." << endl;

        // 初始化 YOLO
        cout << "Initializing YOLO..." << endl;
        ORB_SLAM3::yolov3Segment yolo;
        cout << "YOLO initialized." << endl;

        cv::VideoCapture cap(0);  // 使用 USB 相機

        // 檢查相機是否打開
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera!" << endl;
            return -1;
        }

        cout << "Camera opened successfully." << endl;

        auto start = chrono::system_clock::now();

        // 打開文件來儲存 TUM 格式的位姿數據
        ofstream tumFile;
        tumFile.open(datasetPath + "slam_poses.txt");
        if (!tumFile.is_open()) {
            cerr << "Error: Could not open TUM file!" << endl;
            return -1;
        }

        int frameCount = 0;

        while (true) {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty()) {
                cerr << "Error: Could not grab a frame!" << endl;
                break;
            }

            // 獲取當前時間戳
            auto now = chrono::system_clock::now();
            auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start).count() / 1000.0;

            // 傳遞原始影像給 SLAM 系統，並獲取當前位姿
            cout << "Tracking frame with SLAM..." << endl;
            cv::Mat Tcw = SLAM.TrackMonocular(frame, timestamp);  // 獲取 SLAM 系統的位姿矩陣 (4x4)
            cout << "Frame tracked with SLAM." << endl;

            if (!Tcw.empty()) {
                // 提取平移向量 (tx, ty, tz)
                Eigen::Vector3f translation(Tcw.at<float>(0, 3), Tcw.at<float>(1, 3), Tcw.at<float>(2, 3));

                // 提取旋轉矩陣並轉換為四元數
                Eigen::Matrix3f rotation;
                rotation << Tcw.at<float>(0, 0), Tcw.at<float>(0, 1), Tcw.at<float>(0, 2),
                            Tcw.at<float>(1, 0), Tcw.at<float>(1, 1), Tcw.at<float>(1, 2),
                            Tcw.at<float>(2, 0), Tcw.at<float>(2, 1), Tcw.at<float>(2, 2);

                Eigen::Quaternionf quaternion(rotation);

                // 將位姿數據寫入 TUM 文件
                tumFile << fixed << setprecision(6)
                        << timestamp << " "
                        << translation.x() << " " << translation.y() << " " << translation.z() << " "
                        << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w() << endl;
            }

            // YOLO 部分保持不變
            // 創建一個縮放的副本給 YOLO 使用
            cv::Mat frame_resized;
            cv::resize(frame, frame_resized, cv::Size(512, 512));  // 512x512 用於 YOLO

            // YOLO進行物件檢測並獲得檢測結果
            cout << "Running YOLO segmentation..." << endl;
            auto detectionResult = yolo.Segmentation(frame_resized);
            cout << "YOLO segmentation completed." << endl;

            // 顯示YOLO檢測結果
            cv::Mat yolo_mask = detectionResult.first;
            vector<vector<int>> detections = detectionResult.second;

            // 顯示檢測結果，繪製BoundingBox
            for (const auto &box : detections) {
                if (box.size() < 5) continue;  // 確保box包含足夠的信息
                int classId = box[0];
                int left = box[1], top = box[2], right = box[3], bottom = box[4];
                yolo.drawPred(classId, 1.0, left, top, right, bottom, frame_resized);  // 繪製框
            }

            // 顯示 YOLO 檢測的結果
            cv::imshow("YOLO Detections", frame_resized);

            // 如果按下 'q'，則退出
            if (cv::waitKey(1) == 'q') {
                cout << "Exiting loop." << endl;
                break;
            }

            frameCount++;
        }

        cout << "Shutting down SLAM..." << endl;
        SLAM.Shutdown();  // 關閉 SLAM 系統
        cout << "SLAM shut down." << endl;

        tumFile.close();  // 關閉 TUM 文件
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV error: " << e.what() << endl;
    }
    catch (const std::exception& e) {
        cerr << "Standard exception: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown exception occurred." << endl;
    }

    return 0;
}