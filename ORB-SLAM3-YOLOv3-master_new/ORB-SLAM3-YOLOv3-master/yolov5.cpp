#include "yolov5.h"

namespace ORB_SLAM3 {

    yolov5::yolov5(const std::string& model_path, const std::string& names_path) : start_time(clock()) {
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

    std::vector<torch::Tensor> yolov5::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh) {
        std::vector<torch::Tensor> output;
        for (size_t i = 0; i < preds.sizes()[0]; ++i) {
            torch::Tensor pred = preds.select(0, i);

            // 根據分數過濾
            torch::Tensor scores = pred.select(1, 4) * torch::max(pred.slice(1, 5, pred.sizes()[1]), 1).values();
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

    void yolov5::detect(cv::Mat& frame, cv::Mat& output_frame) {
        // 準備輸入張量
        cv::Mat img;
        cv::resize(frame, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).clone();
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
                if (classID >= 0 && classID < classnames.size()) {
                    cv::putText(output_frame,
                        classnames[classID] + ": " + cv::format("%.2f", score),
                        cv::Point(left, top - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // 計算並顯示 FPS
        clock_t end_time = clock();
        double fps = 1.0 / ((double)(end_time - start_time) / CLOCKS_PER_SEC);
        start_time = end_time;
        cv::putText(output_frame, "FPS: " + std::to_string(int(fps)),
            cv::Point(50, 50),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

}