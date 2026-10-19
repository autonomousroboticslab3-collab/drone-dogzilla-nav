#include <opencv2/opencv.hpp>
#include <torch/script.h> // LibTorch
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>

namespace ORB_SLAM3 {

    class yolov5 {
    public:
        // 建構函數：初始化 YOLOv5 模型和類別名稱
        yolov5(const std::string& model_path, const std::string& names_path) {
            try {
                std::cout << "Loading YOLOv5 model from " << model_path << "..." << std::endl;
                module = torch::jit::load(model_path);
                module.eval();
                std::cout << "YOLOv5 model loaded successfully." << std::endl;

                // 讀取類別名稱
                std::ifstream f(names_path);
                std::string name;
                while (std::getline(f, name)) {
                    classnames.push_back(name);
                }
                if (classnames.empty()) {
                    throw std::runtime_error("Failed to load class names from " + names_path);
                }
                std::cout << "Loaded " << classnames.size() << " class names." << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "Error loading the YOLOv5 model: " << e.what() << std::endl;
                throw e;
            }
            catch (const std::exception& e) {
                std::cerr << "Exception: " << e.what() << std::endl;
                throw e;
            }
        }

        // 非極大值抑制 (Non-Max Suppression)
        std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5) {
            std::vector<torch::Tensor> output;
            for (size_t i = 0; i < preds.sizes()[0]; ++i) {
                torch::Tensor pred = preds.select(0, i);

                // 根據分數過濾
                auto max_result = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
                torch::Tensor scores = pred.select(1, 4) * std::get<0>(max_result);
                pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
                if (pred.sizes()[0] == 0) continue;

                // 將 (center_x, center_y, w, h) 轉換為 (left, top, right, bottom)
                pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
                pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
                pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
                pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

                // 計算分數和類別
                auto max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
                pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
                pred.select(1, 5) = std::get<1>(max_tuple);

                torch::Tensor dets = pred.slice(1, 0, 6);

                torch::Tensor keep = torch::empty({dets.sizes()[0]}, torch::kInt64);
                torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
                auto indexes_tuple = torch::sort(dets.select(1, 4), 0, true); // 降序排序
                torch::Tensor v = std::get<0>(indexes_tuple);
                torch::Tensor indexes = std::get<1>(indexes_tuple);
                int count = 0;
                while (indexes.sizes()[0] > 0) {
                    keep[count] = indexes[0].item().toInt();
                    count += 1;

                    if (indexes.sizes()[0] == 1) break;

                    // 計算重疊部分
                    torch::Tensor lefts = torch::maximum(dets.index({indexes[0].item().toInt(), 0}),
                                                        dets.index({indexes.slice(0, 1, indexes.sizes()[0]), 0}));
                    torch::Tensor tops = torch::maximum(dets.index({indexes[0].item().toInt(), 1}),
                                                       dets.index({indexes.slice(0, 1, indexes.sizes()[0]), 1}));
                    torch::Tensor rights = torch::minimum(dets.index({indexes[0].item().toInt(), 2}),
                                                         dets.index({indexes.slice(0, 1, indexes.sizes()[0]), 2}));
                    torch::Tensor bottoms = torch::minimum(dets.index({indexes[0].item().toInt(), 3}),
                                                          dets.index({indexes.slice(0, 1, indexes.sizes()[0]), 3}));

                    torch::Tensor widths = torch::maximum(torch::zeros_like(rights), rights - lefts);
                    torch::Tensor heights = torch::maximum(torch::zeros_like(bottoms), bottoms - tops);

                    torch::Tensor overlaps = widths * heights;

                    // 計算交集 / 聯集
                    torch::Tensor ious = overlaps / (areas.index({indexes[0].item().toInt()}) +
                                                    torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);

                    // 保留 IOU <= iou_thresh 的索引
                    torch::Tensor mask = (ious <= iou_thresh).nonzero().select(1, 0);
                    indexes = torch::index_select(indexes, 0, mask + 1);
                }
                keep = keep.slice(0, 0, count).toType(torch::kInt64);
                output.push_back(torch::index_select(dets, 0, keep));
            }
            return output;
        }

        // 執行偵測並在影像上繪製結果
        void detect(cv::Mat& frame, cv::Mat& output_frame) {
            // 準備輸入張量
            cv::Mat img;
            cv::resize(frame, img, cv::Size(640, 384));
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
            imgTensor = imgTensor.permute({2, 0, 1});
            imgTensor = imgTensor.toType(torch::kFloat);
            imgTensor = imgTensor.div(255);
            imgTensor = imgTensor.unsqueeze(0);

            // 前向傳播獲得預測結果
            torch::Tensor preds = module.forward({imgTensor}).toTuple()->elements()[0].toTensor();
            std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.4, 0.5);
            output_frame = frame.clone();
            if (dets.size() > 0) {
                // 繪製偵測結果
                for (size_t i = 0; i < dets[0].sizes()[0]; ++i) {
                    float left = dets[0][i][0].item().toFloat() * frame.cols / 640;
                    float top = dets[0][i][1].item().toFloat() * frame.rows / 384;
                    float right = dets[0][i][2].item().toFloat() * frame.cols / 640;
                    float bottom = dets[0][i][3].item().toFloat() * frame.rows / 384;
                    float score = dets[0][i][4].item().toFloat();
                    int classID = dets[0][i][5].item().toInt();

                    // 繪製矩形框
                    cv::rectangle(output_frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);

                    // 顯示類別名稱和分數
                    cv::putText(output_frame,
                        classnames[classID] + ": " + cv::format("%.2f", score),
                        cv::Point(left, top),
                        cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 255, 0), 2);
                }
            }

            // 計算並顯示 FPS
            cv::putText(output_frame, "FPS: " + std::to_string(int(1e7 / (clock() - start_time))),
                cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

    private:
        torch::jit::script::Module module;
        std::vector<std::string> classnames;
        clock_t start_time = clock();
    };

    // 假設 ORB_SLAM3::System 已正確定義，並具有 TrackMonocularyolo 方法
    class System {
    public:
        enum TrackingType { MONOCULAR };
        System(const std::string& vocab_file, const std::string& settings_file, TrackingType type, bool use_viewer) {
            // 初始化 SLAM 系統
            std::cout << "Initializing ORB-SLAM3 with vocab file: " << vocab_file
                      << " and settings file: " << settings_file << std::endl;
            // 此處應初始化 SLAM 系統相關資源
        }

        // 假設存在此方法，用於追踪單目相機並結合 YOLO 檢測結果
        void TrackMonocularyolo(cv::Mat& im, cv::Mat& mask, double timestamp, std::vector<std::vector<int>> detections) {
            // 實現 SLAM 追踪並使用 mask 和 detections
            // 這裡僅為示範，實際應用中需要結合 SLAM 系統的具體實現
        }

        void Shutdown() {
            // 關閉 SLAM 系統
            std::cout << "Shutting down ORB-SLAM3." << std::endl;
        }

        void SaveTrajectoryTUM(const std::string& filename) {
            // 保存相機軌跡到 TUM 格式文件
            std::cout << "Saving trajectory to " << filename << std::endl;
        }

        void SaveKeyFrameTrajectoryTUM(const std::string& filename) {
            // 保存關鍵幀軌跡到 TUM 格式文件
            std::cout << "Saving keyframe trajectory to " << filename << std::endl;
        }
    };

}

int main_yolov5_backup(int argc, char **argv)
{


    // 硬編碼路徑設定（根據實際情況修改）
    std::string vocFile = "/home/t1204/Desktop/ORB-SLAM3-YOLOv3-master/Vocabulary/ORBvoc.txt";// e.g., "/path/to/ORBvoc.txt"
    std::string parameterFile = "/home/t1204/Desktop/ORB-SLAM3-YOLOv3-master/Examples/Monocular/dji.yaml"; // e.g., "/path/to/settings.yaml"
    std::string yoloModelPath = "src/yolo/yolov5s.onnx"; // e.g., "./yolov5s.torchscript.pt"
    std::string yoloNamesPath = "src/yolo/coco.names"; // e.g., "./coco.names"

    // 初始化 YOLO
    ORB_SLAM3::yolov5* yolo = nullptr;
    try {
        std::cout << "Initializing YOLOv5..." << std::endl;
        yolo = new ORB_SLAM3::yolov5(yoloModelPath, yoloNamesPath);
        std::cout << "YOLOv5 initialized successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception while initializing YOLOv5: " << e.what() << std::endl;
        return -1;
    }

    // 初始化 ORB-SLAM3
    ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);
    std::cout << "[Main] ORB-SLAM3 initialized." << std::endl;

    // 打開相機（使用預設相機，通常是索引 0）
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
    {
        std::cerr << "[Main] Error: Could not open the camera!" << std::endl;
        return -1;
    }
    std::cout << "[Main] Camera opened successfully." << std::endl;

    // 打開文件來儲存 TUM 格式的位姿數據
    std::string outputDir = "./"; // 可以根據需要修改輸出目錄
    std::ofstream tumFile;
    std::string tumFilePath = outputDir + "slam_poses.txt";
    tumFile.open(tumFilePath);
    if (!tumFile.is_open()) {
        std::cerr << "[Main] Error: Could not open TUM file at " << tumFilePath << "!" << std::endl;
        return -1;
    }
    std::cout << "[Main] TUM file opened at " << tumFilePath << "." << std::endl;

    // 主迴圈
    cv::Mat im;

    // 膨脹設置（如果需要，可以使用）
    int dilation_size = 15;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                              cv::Point(dilation_size, dilation_size));

    // 記錄開始時間
    auto start = std::chrono::steady_clock::now();

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing camera stream ..." << std::endl << std::endl;

    while(true)
    {
        // 捕捉影像
        if(!cap.read(im))
        {
            std::cerr << "[Main] Error: Could not read frame from camera." << std::endl;
            break;
        }

        // 獲取時間戳（以秒為單位）
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        double tframe = elapsed.count();

        if(im.empty())
        {
            std::cerr << "[Main] Warning: Captured empty frame." << std::endl;
            continue;
        }

        // YOLOv5 偵測
        cv::Mat yolo_output;
        yolo->detect(im, yolo_output);

        // 假設 mask 是全 1，實際應用中應根據 YOLO 的分割結果來設置
        cv::Mat mask = cv::Mat::ones(im.rows, im.cols, CV_8U);
        // TODO: 根據實際需求填充 detections
        std::vector<std::vector<int>> detections; // 這裡應根據 YOLO 的結果來填充

        // 將影像傳遞給 SLAM 系統
        SLAM.TrackMonocularyolo(im, mask, tframe, detections); // 確保 SLAM 類別具有此方法

        // TODO: 將位姿數據寫入 TUM 文件
        // 這裡假設 SLAM 系統有方法來獲取當前位姿
        // cv::Mat Tcw = SLAM.GetCurrentPose(); // 假設有此方法
        // if (!Tcw.empty()) {
        //     // 提取平移和旋轉，並寫入 tumFile
        // }

        // 顯示 YOLO 檢測結果
        cv::imshow("YOLOv5 Detection", yolo_output);

        // 檢查是否按下 'ESC' 鍵來退出
        if(cv::waitKey(1) == 27) break;
    }

    // 停止所有線程
    SLAM.Shutdown();

    // 保存軌跡
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // 釋放 YOLO 物件
    if(yolo != nullptr)
        delete yolo;

    // 關閉 TUM 文件
    tumFile.close();
    std::cout << "[Main] TUM file closed." << std::endl;

    return 0;
}
