#include "EFDEYolov11.h"

// 注册模块
REGISTER_MODULE("MultiModalFusion", EFDEYolo11, EFDEYolo11)

EFDEYolo11::EFDEYolo11(const std::string& exe_path) : IBaseModule(exe_path) 
{
    // 构造函数初始化
    input_buffers_.resize(3, nullptr);
    output_buffers_.resize(1, nullptr);

    // 初始化CUDA流
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("创建CUDA流失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
}

EFDEYolo11::~EFDEYolo11() {
    cleanup();
}

bool EFDEYolo11::init(void* p_pAlgParam) 
{   
    LOG(INFO) << "EFDEYolo11::init status: start ";
    // 1. 配置参数核验
    if (!p_pAlgParam) return false;
    m_config = *static_cast<multimodalfusion::MultiModalFusionModelConfig*>(p_pAlgParam);

    // 2. 配置参数获取
    engine_path_ = m_config.engine_path();
    conf_thres_ = m_config.conf_thres();
    iou_thres_ = m_config.iou_thres();
    num_classes_ = m_config.num_class();
    new_unpad_h_ = m_config.new_unpad_h_rgb();
    new_unpad_w_ = m_config.new_unpad_w_rgb();
    dw_ = m_config.dw_rgb();
    dh_ = m_config.dh_rgb();
    ratio_ = m_config.resize_ratio_rgb();
    stride_.assign(m_config.stride().begin(), m_config.stride().end());
    channels_ = m_config.channels();
    batch_size_ = m_config.batch_size();
    max_dets_ = m_config.max_dets();
    src_width_ = m_config.src_width_rgb();
    src_height_ = m_config.src_height_rgb();
    status_ = m_config.run_status();
    target_size_ = m_config.width();

    // 计算anchor_nums
    for(int stride : stride_) {
        num_anchors_ += (target_size_ / stride) * (target_size_ / stride);
        // LOG(INFO) << "stride = " << stride;
    }
    // LOG(INFO) << "conf_thres_ = " << conf_thres_ << ", iou_thres_ = " << iou_thres_ << ", num_classes_ = " << num_classes_ ", max_dets_ = " << max_dets_;
    // LOG(INFO) << "channels_ = " << channels_ << ", batch_size_ = " << batch_size_;
    // LOG(INFO) << "src_width_ = " << src_width_ << ", src_height_ = " << src_height_;
    // LOG(INFO) << "new_unpad_h_ = " << new_unpad_h_ << ", new_unpad_w_ = " << new_unpad_w_;
    // LOG(INFO) << "dw_ = " << dw_ << ", dh_ = " << dh_;
    // LOG(INFO) << "ratio_ = " << ratio_;
    LOG(INFO) << "num_anchors_ = " << num_anchors_;

    // 初始化TensorRT相关配置
    initTensorRT();
    LOG(INFO) << "EFDEYolo11::init status: success ";
    return true;
}

void EFDEYolo11::initTensorRT() 
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
    // 获取三个输入张量名
    const char* input_name_rgb = engine_->getIOTensorName(0); // images
    const char* input_name_ir = engine_->getIOTensorName(1);  // images2
    const char* input_name_homo = engine_->getIOTensorName(2); // extrinsics
    input_name_ = input_name_rgb;
    input_dims_.nbDims = 4;
    input_dims_.d[0] = batch_size_;
    input_dims_.d[1] = channels_;
    input_dims_.d[2] = target_size_;
    input_dims_.d[3] = target_size_;
    if (!context_->setInputShape(input_name_rgb, input_dims_)) {
        throw std::runtime_error("设置可见光输入形状失败");
    }
    if (!context_->setInputShape(input_name_ir, input_dims_)) {
        throw std::runtime_error("设置红外输入形状失败");
    }
    nvinfer1::Dims homo_dims;
    homo_dims.nbDims = 3;
    homo_dims.d[0] = batch_size_;
    homo_dims.d[1] = 3;
    homo_dims.d[2] = 3;
    if (!context_->setInputShape(input_name_homo, homo_dims)) {
        throw std::runtime_error("设置映射矩阵输入形状失败");
    }
    output_name_ = engine_->getIOTensorName(3); // output
    nvinfer1::Dims out_dims = engine_->getTensorShape(output_name_);
    LOG(INFO) << "engine_->getTensorShape: nbDims=" << out_dims.nbDims;
    for (int i = 0; i < out_dims.nbDims; ++i) {
        LOG(INFO) << "  dim[" << i << "] = " << out_dims.d[i];
    }

    // 5. 分配输入输出buffer
    size_t img_size = batch_size_ * channels_ * target_size_ * target_size_ * sizeof(float);
    size_t homo_size = batch_size_ * 9 * sizeof(float);
    input_buffers_.resize(3, nullptr);
    cudaError_t cuda_status = cudaMalloc(&input_buffers_[0], img_size); // RGB
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("分配可见光输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cuda_status = cudaMalloc(&input_buffers_[1], img_size); // IR
    if (cuda_status != cudaSuccess) {
        cudaFree(input_buffers_[0]);
        throw std::runtime_error("分配红外输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cuda_status = cudaMalloc(&input_buffers_[2], homo_size); // Homography
    if (cuda_status != cudaSuccess) {
        cudaFree(input_buffers_[0]);
        cudaFree(input_buffers_[1]);
        throw std::runtime_error("分配映射矩阵输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    // 输出buffer
    size_t output_size = batch_size_ * (4 + num_classes_) * num_anchors_ * sizeof(float);
    LOG(INFO) << "init TensorRT : output_size = " << output_size << ", num_anchors_ = " << num_anchors_ << "num_classes_ = " << num_classes_;
    void* output_buffer = nullptr;
    cuda_status = cudaMalloc(&output_buffer, output_size);
    if (cuda_status != cudaSuccess) {
        cudaFree(input_buffers_[0]);
        cudaFree(input_buffers_[1]);
        cudaFree(input_buffers_[2]);
        throw std::runtime_error("分配输出GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    output_buffers_[0] = output_buffer;

    // 6. 绑定输入输出buffer
    if (!context_->setTensorAddress(input_name_rgb, input_buffers_[0])) {
        cudaFree(input_buffers_[0]);
        cudaFree(input_buffers_[1]);
        cudaFree(input_buffers_[2]);
        cudaFree(output_buffer);
        throw std::runtime_error("绑定可见光输入张量失败");
    }
    if (!context_->setTensorAddress(input_name_ir, input_buffers_[1])) {
        cudaFree(input_buffers_[0]);
        cudaFree(input_buffers_[1]);
        cudaFree(input_buffers_[2]);
        cudaFree(output_buffer);
        throw std::runtime_error("绑定红外输入张量失败");
    }
    if (!context_->setTensorAddress(input_name_homo, input_buffers_[2])) {
        cudaFree(input_buffers_[0]);
        cudaFree(input_buffers_[1]);
        cudaFree(input_buffers_[2]);
        cudaFree(output_buffer);
        throw std::runtime_error("绑定映射矩阵输入张量失败");
    }
    if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
        cudaFree(input_buffers_[0]);
        cudaFree(input_buffers_[1]);
        cudaFree(input_buffers_[2]);
        cudaFree(output_buffer);
        throw std::runtime_error("绑定输出张量失败");
    }
    LOG(INFO) << "已绑定所有输入张量: " << input_name_rgb << ", " << input_name_ir << ", " << input_name_homo;
}

void EFDEYolo11::setInput(void* input) 
{   
    // 核验输入数据的合法性并进行类型转换和保存
    if (!input) {
        LOG(ERROR) << "输入为空";
        return;
    }
    auto* vec = static_cast<std::vector<std::vector<float>>*>(input);
    if (vec->size() != 3) {
        LOG(ERROR) << "输入vector数量错误，期望3，实际" << vec->size();
        return;
    }
    size_t img_len = batch_size_ * channels_ * target_size_ * target_size_;
    if ((*vec)[0].size() != img_len) {
        LOG(ERROR) << "可见光数据长度错误，期望" << img_len << "，实际" << (*vec)[0].size();
        return;
    }
    if ((*vec)[1].size() != img_len) {
        LOG(ERROR) << "红外数据长度错误，期望" << img_len << "，实际" << (*vec)[1].size();
        return;
    }
    if ((*vec)[2].size() != 9) {
        LOG(ERROR) << "映射矩阵长度错误，期望9，实际" << (*vec)[2].size();
        return;
    }
    m_inputImage = *vec;
}

void* EFDEYolo11::getOutput() {
    return &m_outputResult;
}

void EFDEYolo11::execute() 
{
    // 1、执行推理
    std::vector<float> output = inference();

    // 2、结果后处理
    std::vector<std::vector<float>> results = process_output(output);

    // 3、输出格式转换
    m_outputResult = formatConverted(results);
}


void EFDEYolo11::cleanup() 
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
CAlgResult EFDEYolo11::formatConverted(std::vector<std::vector<float>> results)
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
        obj_result.strClass(std::to_string(static_cast<int>(result[5])));
        frame_result.vecObjectResult().push_back(obj_result);
    }
    alg_result.vecFrameResult({frame_result});
    return alg_result;
}

std::vector<float> EFDEYolo11::inference()
{
    LOG(INFO) << "EFDEYolo11::inference status: start ";
    // LOG(INFO) << "推理输入 shape: " << batch_size_ << "x" << channels_ << "x" << new_unpad_h_ << "x" << new_unpad_w_;

    // 1. 设置动态输入尺寸
    input_dims_.nbDims = 4;
    input_dims_.d[0] = batch_size_;  // batch size
    input_dims_.d[1] = channels_;  // channels
    input_dims_.d[2] = target_size_;
    input_dims_.d[3] = target_size_;
    size_t img_size = batch_size_ * channels_ * target_size_ * target_size_;
    // 校验输入
    if (m_inputImage.size() != 3) {
        LOG(ERROR) << "m_inputImage.size() != 3";
        return {};
    }
    if (m_inputImage[0].size() != img_size) {
        LOG(ERROR) << "可见光数据长度错误，期望" << img_size << "，实际" << m_inputImage[0].size();
        return {};
    }
    if (m_inputImage[1].size() != img_size) {
        LOG(ERROR) << "红外数据长度错误，期望" << img_size << "，实际" << m_inputImage[1].size();
        return {};
    }
    size_t batch_homo33 = batch_size_ * 3 * 3;
    if (m_inputImage[2].size() != batch_homo33) {
        LOG(ERROR) << "映射矩阵长度错误，必须为batch*3*3=" << batch_homo33 << "，实际" << m_inputImage[2].size();
        return {};
    }
    // LOG(INFO) << "推理输入 shape: batch=" << bds) << ", 红外长度: " << m_inputImage[1].size() << ", 映射矩阵长度: " << m_inputImage[2].size();

    // 2. 绑定输入输出 buffer
    // 多输入：每个输入都要绑定
    const char* input_name_rgb = engine_->getIOTensorName(0);
    const char* input_name_ir = engine_->getIOTensorName(1);
    const char* input_name_homo = engine_->getIOTensorName(2);
    if (!context_->setTensorAddress(input_name_rgb, input_buffers_[0])) {
        throw std::runtime_error("绑定可见光输入张量失败");
    }
    if (!context_->setTensorAddress(input_name_ir, input_buffers_[1])) {
        throw std::runtime_error("绑定红外输入张量失败");
    }
    if (!context_->setTensorAddress(input_name_homo, input_buffers_[2])) {
        throw std::runtime_error("绑定映射矩阵输入张量失败");
    }
    if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
        throw std::runtime_error("设置输出张量地址失败");
    }
    // LOG(INFO) << "已绑定所有输入张量: " << input_name_rgb << ", " << input_name_ir << ", " << input_name_homo;

    // 3. 拷贝输入数据到GPU
    cudaError_t cuda_status = cudaMemcpyAsync(input_buffers_[0], m_inputImage[0].data(),
                                              m_inputImage[0].size() * sizeof(float),
                                              cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "可见光 CUDA内存拷贝失败: " << cudaGetErrorString(cuda_status);
        return {};
    }
    cuda_status = cudaMemcpyAsync(input_buffers_[1], m_inputImage[1].data(),
                                  m_inputImage[1].size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "红外 CUDA内存拷贝失败: " << cudaGetErrorString(cuda_status);
        return {};
    }
    cuda_status = cudaMemcpyAsync(input_buffers_[2], m_inputImage[2].data(),
                                  m_inputImage[2].size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "映射矩阵 CUDA内存拷贝失败: " << cudaGetErrorString(cuda_status);
        return {};
    }
    cudaStreamSynchronize(stream_);

    // 4. 执行推理
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        LOG(ERROR) << "TensorRT推理失败";
        return {};
    }
    cudaStreamSynchronize(stream_);

    // 5. 获取输出 shape
    nvinfer1::Dims out_dims2 = context_->getTensorShape(output_name_);
    // LOG(INFO) << "context_->getTensorShape: nbDims=" << out_dims2.nbDims;
    // for (int i = 0; i < out_dims2.nbDims; ++i) {
    //     LOG(INFO) << "  dim[" << i << "] = " << out_dims2.d[i];
    // }
    size_t output_size = 1;
    for (int i = 0; i < out_dims2.nbDims; ++i) {
        output_size *= out_dims2.d[i];
    }
    std::vector<float> output(output_size);

    // 6. 拷贝输出数据到CPU
    cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                  output_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_);
    // LOG(INFO) << "output_size = " << output_size;
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "CUDA输出内存拷贝失败: " << cudaGetErrorString(cuda_status);
        return {};
    }
    cudaStreamSynchronize(stream_);

    // 7. 保存推理输出为bin文件
    // if (status_) {
    //     save_bin(output, "./Save_Data/multimodal/result/output_EFDEYolo11.bin"); // EFDEYolo11/Inference
    // }

    // LOG(INFO) << "推理输出 shape: " << output.size();
    LOG(INFO) << "EFDEYolo11::inference status: success ";
    return output;
}

std::vector<std::vector<float>> EFDEYolo11::process_output(const std::vector<float>& output)
{
    // 1. TensorRT输出数据转置
    int num_anchors = num_anchors_;
    int feature_dim = 4 + num_classes_;
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
        // std::cout << "x: " << x << ", y: " << y << ", w: " << w << ", h: " << h << std::endl;
        float x1 = (x - w / 2 - dw_) / ratio_;
        float y1 = (y - h / 2 - dh_) / ratio_;
        float x2 = (x + w / 2 - dw_) / ratio_;
        float y2 = (y + h / 2 - dh_) / ratio_;
        // std::cout << "x1: " << x1 << ", y1: " << y1 << ", x2: " << x2 << ", y2: " << y2 << std::endl;

        boxes.push_back({x1, y1, x2, y2});
        scores.push_back(max_conf);
        class_ids.push_back(max_class);
    }

    // std::cout << "boxes.size(): " << boxes.size() << std::endl;
    // std::cout << "scores.size(): " << scores.size() << std::endl;
    // std::cout << "class_ids.size(): " << class_ids.size() << std::endl;

    // 3. 按类别分组做NMS
    for (int cls = 0; cls < num_classes_; ++cls) {
        std::vector<std::vector<float>> cls_boxes;
        std::vector<float> cls_scores;
        for (size_t i = 0; i < class_ids.size(); ++i) {
            if (class_ids[i] == cls) {
                cls_boxes.push_back(boxes[i]);
                cls_scores.push_back(scores[i]);
            }
        }
        if (cls_boxes.empty()) continue;
        std::vector<int> keep = nms(cls_boxes, cls_scores);
        for (int idx : keep) {
            std::vector<float> result = cls_boxes[idx];
            result.push_back(cls_scores[idx]);
            result.push_back(static_cast<float>(cls));
            results.push_back(result);
        }
    }

    // 全局NMS
    std::vector<float> global_scores;
    for (const auto& res : results) {
        global_scores.push_back(res[4]); // 置信度
    }
    std::vector<int> keep = nms(results, global_scores); // 用现有nms函数
    std::vector<std::vector<float>> final_results;
    for (int idx : keep) {
        final_results.push_back(results[idx]);
    }

    // 排序和截断
    std::sort(final_results.begin(), final_results.end(), [](const std::vector<float>& a, const std::vector<float>& b) {
        return a[4] > b[4];
    });
    if (final_results.size() > max_dets_) final_results.resize(max_dets_);

    // if(status_)
    // {
    //     save_bin(results, "./Save_Data/multimodal/result/processed_output_EFDEYolo11.bin"); // EFDEYolo11/Inference
    // }
    return final_results;
}

std::vector<int> EFDEYolo11::nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores) 
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