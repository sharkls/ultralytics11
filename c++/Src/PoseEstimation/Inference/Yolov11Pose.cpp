#include "Yolov11Pose.h"


// #include <NvInfer.h>
// #include <cuda_runtime_api.h>


Yolov11Pose::Yolov11Pose(const std::string& exe_path) : IBaseModule(exe_path) {
    // 构造函数初始化
    input_buffers_.resize(1, nullptr);
    output_buffers_.resize(1, nullptr);
    // 其它成员初始化
}

Yolov11Pose::~Yolov11Pose() {
    cleanup();
}

bool Yolov11Pose::init(void* p_pAlgParam) {
    if (!p_pAlgParam) return false;
    m_poseConfig = *static_cast<PoseConfig*>(p_pAlgParam);

    const auto& yolo_cfg = m_poseConfig.yolo_model_config();
    conf_thres_ = yolo_cfg.conf_thres();
    iou_thres_ = yolo_cfg.iou_thres();
    num_classes_ = yolo_cfg.num_class();
    new_unpad_h_ = yolo_cfg.height();
    new_unpad_w_ = yolo_cfg.width();
    // dw/dh/ratio/stride 需根据实际预处理逻辑设置，这里假设proto有这些字段
    dw_ = yolo_cfg.dw();
    dh_ = yolo_cfg.dh();
    ratio_ = yolo_cfg.resize_ratio();
    stride_ = yolo_cfg.stride();
    batch_size_ = yolo_cfg.batch_size();


    // 这里初始化TensorRT引擎、分配buffer等，参考trt_infer.cpp
    // ... engine加载、runtime/context初始化、buffer分配等 ...
    return true;
}

void Yolov11Pose::setInput(void* input) {
    if (!input) {
        LOG(ERROR) << "输入为空";
        return;
    }
    m_inputImage = *static_cast<cv::Mat*>(input);
}

void* Yolov11Pose::getOutput() {
    return &m_outputResult;
}

void* Yolov11Pose::execute() {
    if (m_inputImage.empty()) {
        LOG(ERROR) << "预处理输出为空";
        return nullptr;
    }
    cv::Mat preprocessed = m_inputImage;

    const auto& yolo_cfg = m_poseConfig.yolo_model_config();
    std::vector<float> output = inference(preprocessed);

    

    auto results = process_output(output);
    // m_outputResult = ...;
    return &m_outputResult;
}

void Yolov11Pose::cleanup() {
    for (auto& buf : input_buffers_) {
        if (buf) {/* cudaFree(buf); */}
    }
    for (auto& buf : output_buffers_) {
        if (buf) {/* cudaFree(buf); */}
    }
    if (stream_) {/* cudaStreamDestroy(stream_); */}
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

std::vector<float> Yolov11Pose::inference(const cv::Mat& img)
{
    std::vector<float> input;

    if (input_buffers_.empty() || input_buffers_[0] == nullptr) {
        throw std::runtime_error("输入缓冲区未正确初始化");
    }

    const char* input_name = engine_->getIOTensorName(0);
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = batch_size_;
    input_dims.d[1] = 3;
    input_dims.d[2] = new_unpad_h_;
    input_dims.d[3] = new_unpad_w_;

    if (!context_->setInputShape(input_name, input_dims)) {
        throw std::runtime_error("设置输入形状失败");
    }

    size_t input_size = input.size() * sizeof(float);
    cudaError_t cuda_status = cudaMemcpyAsync(input_buffers_[0], input.data(),
                                              input_size,
                                              cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }

    bool status = context_->enqueueV3(stream_);
    if (!status) {
        throw std::runtime_error("TensorRT推理失败");
    }
    cudaStreamSynchronize(stream_);

    const char* output_name = engine_->getIOTensorName(1);
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name);

    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }

    std::vector<float> output(output_size);
    cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                  output_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA输出内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    return output;
}

void Yolov11Pose::preprocess(const cv::Mat& img, std::vector<float>& input) {
    // 参考trt_infer.cpp实现，直接用成员变量
    // ...
}

void Yolov11Pose::rescale_coords(std::vector<float>& coords, bool is_keypoint) {
    if (coords.empty()) return;
    if (is_keypoint) {
        for (size_t i = 0; i < coords.size(); i += 3) {
            coords[i] = (coords[i] - dw_) / ratio_;
            coords[i + 1] = (coords[i + 1] - dh_) / ratio_;
        }
    } else {
        for (size_t i = 0; i < coords.size(); i += 4) {
            coords[i] = (coords[i] - dw_) / ratio_;
            coords[i + 1] = (coords[i + 1] - dh_) / ratio_;
            coords[i + 2] = (coords[i + 2] - dw_) / ratio_;
            coords[i + 3] = (coords[i + 3] - dh_) / ratio_;
        }
    }
}

std::vector<std::vector<float>> Yolov11Pose::process_keypoints(const std::vector<float>& output, const std::vector<std::vector<float>>& boxes) {
    std::vector<std::vector<float>> keypoints;
    int num_keypoints = 17; // 可根据模型实际调整
    for (size_t i = 0; i < boxes.size(); ++i) {
        std::vector<float> kpts;
        int kpt_start = 6; // 可根据模型实际调整
        for (int j = 0; j < num_keypoints; ++j) {
            float x = output[kpt_start + j * 3];
            float y = output[kpt_start + j * 3 + 1];
            float conf = output[kpt_start + j * 3 + 2];
            kpts.push_back(x);
            kpts.push_back(y);
            kpts.push_back(conf);
        }
        rescale_coords(kpts, true);
        keypoints.push_back(kpts);
    }
    return keypoints;
}

std::vector<std::vector<float>> Yolov11Pose::process_output(const std::vector<float>& output) {
    // 1. 计算anchor数量和特征维度
    int num_anchors = (new_unpad_h_ / stride_) * (new_unpad_w_ / stride_);
    int feature_dim = 4 + num_classes_ + 51; // 4 bbox + num_classes + 17*3关键点
    std::vector<std::vector<float>> results;

    // 2. 遍历所有anchor，筛选置信度大于阈值的候选框
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> keypoints;

    for (int i = 0; i < num_anchors; ++i) {
        // 取bbox
        float x = output[i * feature_dim + 0];
        float y = output[i * feature_dim + 1];
        float w = output[i * feature_dim + 2];
        float h = output[i * feature_dim + 3];

        // 取类别分数
        float max_conf = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes_; ++c) {
            float conf = output[i * feature_dim + 4 + c];
            if (conf > max_conf) {
                max_conf = conf;
                max_class = c;
            }
        }
        if (max_conf < conf_thres_) continue;

        // 坐标还原
        float x1 = (x - w / 2 - dw_) / ratio_;
        float y1 = (y - h / 2 - dh_) / ratio_;
        float x2 = (x + w / 2 - dw_) / ratio_;
        float y2 = (y + h / 2 - dh_) / ratio_;

        boxes.push_back({x1, y1, x2, y2});
        scores.push_back(max_conf);
        class_ids.push_back(max_class);

        // 关键点
        std::vector<float> kpts;
        for (int j = 0; j < 51; j += 3) {
            float kpt_x = (output[i * feature_dim + 4 + num_classes_ + j] - dw_) / ratio_;
            float kpt_y = (output[i * feature_dim + 4 + num_classes_ + j + 1] - dh_) / ratio_;
            float kpt_conf = output[i * feature_dim + 4 + num_classes_ + j + 2];
            kpts.push_back(kpt_x);
            kpts.push_back(kpt_y);
            kpts.push_back(kpt_conf);
        }
        keypoints.push_back(kpts);
    }

    // 3. 按类别分组做NMS
    for (int cls = 0; cls < num_classes_; ++cls) {
        std::vector<std::vector<float>> cls_boxes;
        std::vector<float> cls_scores;
        std::vector<std::vector<float>> cls_keypoints;
        for (size_t i = 0; i < class_ids.size(); ++i) {
            if (class_ids[i] == cls) {
                cls_boxes.push_back(boxes[i]);
                cls_scores.push_back(scores[i]);
                cls_keypoints.push_back(keypoints[i]);
            }
        }
        if (cls_boxes.empty()) continue;
        std::vector<int> keep = nms(cls_boxes, cls_scores);
        for (int idx : keep) {
            std::vector<float> result = cls_boxes[idx];
            result.push_back(cls_scores[idx]);
            result.push_back(static_cast<float>(cls));
            result.insert(result.end(), cls_keypoints[idx].begin(), cls_keypoints[idx].end());
            results.push_back(result);
        }
    }

    // 4. 按置信度排序，截断最大检测数
    std::sort(results.begin(), results.end(), [](const std::vector<float>& a, const std::vector<float>& b) {
        return a[4] > b[4];
    });
    if (results.size() > 300) results.resize(300);

    return results;
}

std::vector<int> Yolov11Pose::nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores) 
{
    float iou_threshold = 0.5f; // 可根据成员变量或配置调整
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);
        indices.erase(indices.begin());

        std::vector<int> tmp_indices;
        for (int i : indices) {
            float iou = 0.0f;
            float xx1 = std::max(boxes[idx][0], boxes[i][0]);
            float yy1 = std::max(boxes[idx][1], boxes[i][1]);
            float xx2 = std::min(boxes[idx][2], boxes[i][2]);
            float yy2 = std::min(boxes[idx][3], boxes[i][3]);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;

            float area1 = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]);
            float area2 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
            float union_area = area1 + area2 - intersection;

            iou = intersection / (union_area + 1e-16f);

            if (iou <= iou_threshold) {
                tmp_indices.push_back(i);
            }
        }
        indices = tmp_indices;
    }
    return keep;
}