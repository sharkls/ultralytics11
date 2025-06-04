#include "Yolov11Pose.h"

// 注册模块
REGISTER_MODULE("PoseEstimation", Yolov11Pose, Yolov11Pose)

Yolov11Pose::Yolov11Pose(const std::string& exe_path) : IBaseModule(exe_path) 
{
    // 构造函数初始化
    input_buffers_.resize(1, nullptr);
    output_buffers_.resize(1, nullptr);

    // 初始化CUDA流
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("创建CUDA流失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
}

Yolov11Pose::~Yolov11Pose() {
    cleanup();
}

bool Yolov11Pose::init(void* p_pAlgParam) 
{   
    LOG(INFO) << "Yolov11Pose::init status: start ";
    // 1. 配置参数核验
    if (!p_pAlgParam) return false;
    m_poseConfig = *static_cast<posetimation::YOLOModelConfig*>(p_pAlgParam);

    // 2. 配置参数获取
    engine_path_ = m_poseConfig.engine_path();
    conf_thres_ = m_poseConfig.conf_thres();
    iou_thres_ = m_poseConfig.iou_thres();
    num_classes_ = m_poseConfig.num_class();
    new_unpad_h_ = m_poseConfig.new_unpad_h();
    new_unpad_w_ = m_poseConfig.new_unpad_w();
    dw_ = m_poseConfig.dw();
    dh_ = m_poseConfig.dh();
    ratio_ = m_poseConfig.resize_ratio();
    stride_.assign(m_poseConfig.stride().begin(), m_poseConfig.stride().end());
    channels_ = m_poseConfig.channels();
    batch_size_ = m_poseConfig.batch_size();
    num_keys_ = m_poseConfig.num_keys();
    max_dets_ = m_poseConfig.max_dets();
    src_width_ = m_poseConfig.src_width();
    src_height_ = m_poseConfig.src_height();
    status_ = m_poseConfig.run_status();

    // 计算anchor_nums
    for(int stride : stride_) {
        num_anchors_ += (new_unpad_h_ / stride) * (new_unpad_w_ / stride);
        // LOG(INFO) << "stride = " << stride;
    }
    // LOG(INFO) << "conf_thres_ = " << conf_thres_ << ", iou_thres_ = " << iou_thres_ << ", num_classes_ = " << num_classes_ << ", num_keys_ = " << num_keys_ << ", max_dets_ = " << max_dets_;
    // LOG(INFO) << "channels_ = " << channels_ << ", batch_size_ = " << batch_size_;
    // LOG(INFO) << "src_width_ = " << src_width_ << ", src_height_ = " << src_height_;
    // LOG(INFO) << "new_unpad_h_ = " << new_unpad_h_ << ", new_unpad_w_ = " << new_unpad_w_;
    // LOG(INFO) << "dw_ = " << dw_ << ", dh_ = " << dh_;
    // LOG(INFO) << "ratio_ = " << ratio_;
    // LOG(INFO) << "num_anchors_ = " << num_anchors_;

    // 初始化TensorRT相关配置
    initTensorRT();
    LOG(INFO) << "Yolov11Pose::init status: success ";
    return true;
}

void Yolov11Pose::initTensorRT() 
{
    // 初始化TensorRT引擎、分配buffer、engine加载、runtime/context初始化、buffer分配等
    // 1. 加载engine
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("无法打开engine文件");
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    LOG(INFO) << "engine_path_: " << engine_path_;
    LOG(INFO) << "[TensorRT] Loaded engine size: " << size / (1024 * 1024) << " MiB";

    // 2. 创建runtime和engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("创建TensorRT runtime失败");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    if (!engine_) {
        throw std::runtime_error("反序列化engine失败");
    }

    // 3. 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("创建执行上下文失败");
    }

    // 4. 获取输入输出信息
    input_name_ = engine_->getIOTensorName(0);
    input_dims_.nbDims = 4;
    input_dims_.d[0] = batch_size_;
    input_dims_.d[1] = channels_;
    input_dims_.d[2] = new_unpad_h_;
    input_dims_.d[3] = new_unpad_w_;
    if (!context_->setInputShape(input_name_, input_dims_)) {
        throw std::runtime_error("设置输入形状失败");
    }

    output_name_ = engine_->getIOTensorName(1);
    output_dims_ = context_->getTensorShape(output_name_);

    // 5. 计算输入输出大小
    input_size_ = batch_size_ * channels_ * new_unpad_h_ * new_unpad_w_;
    output_size_ = batch_size_ * (4 + num_classes_ + num_keys_ * 3) * num_anchors_;
    LOG(INFO) << "input_size_ = " << input_size_ << ", output_size_ = " << output_size_ << ", num_anchors_ : " << num_anchors_;

    // 6.1 分配输入GPU内存
    void* input_buffer = nullptr;
    cudaError_t cuda_status = cudaMalloc(&input_buffer, input_size_ * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("分配输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    input_buffers_[0] = input_buffer;

    // 6.2 分配输出GPU内存
    void* output_buffer = nullptr;
    cuda_status = cudaMalloc(&output_buffer, output_size_ * sizeof(float));
    if (cuda_status != cudaSuccess) {
        // 清理已分配的内存
        cudaFree(input_buffer);
        throw std::runtime_error("分配输出GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    output_buffers_[0] = output_buffer;

    // 7.设置绑定
    if (!context_->setTensorAddress(input_name_, input_buffers_[0])) {
        // 清理已分配的内存
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("设置输入张量地址失败");
    }
    if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
        // 清理已分配的内存
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("设置输出张量地址失败");
    }
}

void Yolov11Pose::setInput(void* input) 
{   
    // 核验输入数据的合法性并进行类型转换和保存
    if (!input) {
        LOG(ERROR) << "输入为空";
        return;
    }
    m_inputImage = *static_cast<std::vector<float>*>(input);
    // LOG(INFO) << "m_inputImage.size() = " << m_inputImage.size() << ", input_size_ = " << input_size_;
}

void* Yolov11Pose::getOutput() {
    return &m_outputResult;
}

void Yolov11Pose::execute() 
{
    // 1、执行推理
    std::vector<float> output = inference();

    // 2、结果后处理
    std::vector<std::vector<float>> results = process_output(output);

    // 3、输出格式转换
    m_outputResult = formatConverted(results);
    LOG(INFO) << "Yolov11Pose::execute status: success " << m_outputResult.vecFrameResult().size();

}


void Yolov11Pose::cleanup() 
{
    for (auto& buf : input_buffers_) {
        if (buf) { cudaFree(buf); buf = nullptr; }
    }
    for (auto& buf : output_buffers_) {
        if (buf) { cudaFree(buf); buf = nullptr; }
    }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

// 将模型输出结果转换为CAlgResult
CAlgResult Yolov11Pose::formatConverted(std::vector<std::vector<float>> results)
{
    CAlgResult alg_result;
    CFrameResult frame_result;

    for (const auto& result : results) {
        CObjectResult obj_result;

        // 边界框
        obj_result.fTopLeftX(result[0]);
        obj_result.fTopLeftY(result[1]);
        obj_result.fBottomRightX(result[2]);
        obj_result.fBottomRightY(result[3]);
        // 置信度
        obj_result.fVideoConfidence(result[4]);
        // 类别
        // obj_result.strClass(std::to_string(static_cast<int>(result[5]))); // 注释原有类别设置

        // 关键点
        std::vector<Keypoint> keypoints;
        std::vector<float> keypoints_vec;
        for (int j = 0; j < num_keys_; ++j) 
        {
            Keypoint kp;
            float x = result[6 + j * 3];
            float y = result[6 + j * 3 + 1];
            float conf = result[6 + j * 3 + 2];
            kp.x(x);
            kp.y(y);
            kp.confidence(conf);
            keypoints.push_back(kp);
            keypoints_vec.push_back(x);
            keypoints_vec.push_back(y);
            keypoints_vec.push_back(conf);
        }
        obj_result.vecKeypoints(keypoints);

        // --- 新增：根据关键点分类并设置类别 ---
        std::string pose_state = classify_pose(keypoints_vec);
        std::cout << "pose_state: " << pose_state << std::endl;
        if (pose_state == "躺着" || pose_state == "坐着或佝偻" || pose_state == "未知") {
            obj_result.strClass("0");
        } else {
            obj_result.strClass("1");
        }
        // 可选：也可将状态写入日志
        // LOG(INFO) << "Pose state: " << pose_state;
        // --- 新增结束 ---

        frame_result.vecObjectResult().push_back(obj_result);
    }

    alg_result.vecFrameResult({frame_result});

    LOG(INFO) << "formatConverted: alg_result.vecFrameResult().size() = " << alg_result.vecFrameResult().size();
    if (alg_result.vecFrameResult().size() > 0)
        LOG(INFO) << "formatConverted: frame_result.vecObjectResult().size() = " << alg_result.vecFrameResult()[0].vecObjectResult().size();
    return alg_result;
}

std::vector<float> Yolov11Pose::inference()
{
    LOG(INFO) << "Yolov11Pose::inference status: start ";
    // LOG(INFO) << "推理输入 shape: " << batch_size_ << "x" << channels_ << "x" << new_unpad_h_ << "x" << new_unpad_w_;

    // 1. 设置动态输入尺寸
    input_dims_.nbDims = 4;
    input_dims_.d[0] = batch_size_;  // batch size
    input_dims_.d[1] = channels_;  // channels
    input_dims_.d[2] = new_unpad_h_;
    input_dims_.d[3] = new_unpad_w_;
    if (!context_->setInputShape(input_name_, input_dims_)) {
        throw std::runtime_error("设置输入形状失败");
    }

    // 2. 绑定输入输出 buffer
    if (!context_->setTensorAddress(input_name_, input_buffers_[0])) {
        throw std::runtime_error("设置输入张量地址失败");
    }
    if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
        throw std::runtime_error("设置输出张量地址失败");
    }

    // 3. 拷贝输入数据到GPU
    size_t input_size = m_inputImage.size() * sizeof(float);
    cudaError_t cuda_status = cudaMemcpyAsync(input_buffers_[0], m_inputImage.data(),
                                              input_size,
                                              cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    // 4. 执行推理
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        throw std::runtime_error("TensorRT推理失败");
    }
    cudaStreamSynchronize(stream_);

    // 5. 获取输出 shape
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    std::vector<float> output(output_size);

    // 6. 拷贝输出数据到CPU
    cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                  output_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA输出内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    // 7. 保存推理输出为bin文件
    // if (status_) {
    //     save_bin(output, "./Save_Data/pose/result/inference_output_yolov11pose.bin"); // Yolov11Pose/Inference
    // }

    // LOG(INFO) << "推理输出 shape: " << output.size();
    LOG(INFO) << "Yolov11Pose::inference status: success ";
    return output;
}

void Yolov11Pose::rescale_coords(std::vector<float>& coords, bool is_keypoint) 
{
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
    int num_keypoints = num_keys_; // 可根据模型实际调整
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

std::vector<std::vector<float>> Yolov11Pose::process_output(const std::vector<float>& output) 
{   
    LOG(INFO) << "Yolov11Pose::process_output status: start ";
    // 1. TensorRT输出数据转置
    // [batch_size, 4 + num_classes + num_keys * 3, num_anchors] -> [num_anchors, 4 + num_classes + num_keys * 3]
    int num_anchors = num_anchors_; 
    int feature_dim = 4 + num_classes_ + num_keys_ * 3; 
    std::vector<std::vector<float>> results;
    std::vector<float> output_trans(num_anchors * feature_dim);

    for (int i = 0; i < num_anchors; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            output_trans[i * feature_dim + j] = output[j * num_anchors + i];
        }
    }
    
    // 2. 遍历所有anchor，筛选置信度大于阈值的候选框
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> keypoints;
    for (int i = 0; i < num_anchors_; ++i) 
    {
        // 取bbox
        float x = output_trans[i * feature_dim + 0];
        float y = output_trans[i * feature_dim + 1];
        float w = output_trans[i * feature_dim + 2];
        float h = output_trans[i * feature_dim + 3];

        // 取类别分数
        float max_conf = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes_; ++c) {
            float conf = output_trans[i * feature_dim + 4 + c];
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
        for (int j = 0; j < num_keys_ * 3; j += 3) {
            float kpt_x = (output_trans[i * feature_dim + 4 + num_classes_ + j] - dw_) / ratio_;
            float kpt_y = (output_trans[i * feature_dim + 4 + num_classes_ + j + 1] - dh_) / ratio_;
            float kpt_conf = output_trans[i * feature_dim + 4 + num_classes_ + j + 2];
            if (kpt_conf < conf_thres_) {   // 祛除低于置信度阈值的关键点
                kpts.push_back(0.0f);
                kpts.push_back(0.0f);
                kpts.push_back(0.0f);
            } else {
                kpts.push_back(kpt_x);
                kpts.push_back(kpt_y);
                kpts.push_back(kpt_conf);
            }
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
    if (results.size() > max_dets_) results.resize(max_dets_);

    if(status_)
    {
        save_bin(results, "./Save_Data/pose/result/processed_output_yolov11pose.bin"); // Yolov11Pose/Inference
    }
    LOG(INFO) << "Yolov11Pose::process_output status: success ";
    return results;
}

std::vector<int> Yolov11Pose::nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores) 
{
    float iou_threshold = iou_thres_; // 可根据成员变量或配置调整
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

std::string Yolov11Pose::classify_pose(const std::vector<float>& keypoints) const
{
    if (keypoints.size() < num_keys_ * 3) return "未知";
    // std::cout << "keypoints.size(): " << keypoints.size() << std::endl;

    // 关键点下标（COCO格式）
    int LShoulder = 5, RShoulder = 6, LHip = 11, RHip = 12, LKnee = 13, RKnee = 14, Nose = 0;

    auto get_pt = [&](int idx) {
        return cv::Point2f(keypoints[idx * 3], keypoints[idx * 3 + 1]);
    };

    cv::Point2f shoulder_mid = (get_pt(LShoulder) + get_pt(RShoulder)) * 0.5f;
    cv::Point2f hip_mid = (get_pt(LHip) + get_pt(RHip)) * 0.5f;
    cv::Point2f knee_mid = (get_pt(LKnee) + get_pt(RKnee)) * 0.5f;

    // 主轴角度（修正：与y轴夹角，站立为0°，躺着为90°）
    cv::Point2f axis = shoulder_mid - hip_mid;
    float axis_angle = std::atan2(axis.x, axis.y) * 180.0f / CV_PI;
    axis_angle = std::abs(axis_angle);
    if (axis_angle > 90.0f) axis_angle = 180.0f - axis_angle;

    // 躯干夹角
    auto angle = [](cv::Point2f a, cv::Point2f b, cv::Point2f c) {
        cv::Point2f v1 = a - b, v2 = c - b;
        float dot = v1.dot(v2);
        float norm = cv::norm(v1) * cv::norm(v2);
        if (norm < 1e-3f) return 180.0f;
        float cos_theta = dot / norm;
        cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
        return static_cast<float>(std::acos(cos_theta) * 180.0 / CV_PI);
    };
    float trunk_angle = angle(shoulder_mid, hip_mid, knee_mid);

    // std::cout << "axis_angle: " << axis_angle << std::endl;
    // std::cout << "trunk_angle: " << trunk_angle << std::endl;

    // 判断
    if (axis_angle < 30 && trunk_angle > 150)
        return "站立/行走";
    else if (axis_angle > 60)
        return "躺着";
    else if (trunk_angle < 120)
        return "坐着或佝偻";
    else
        return "未知";
}