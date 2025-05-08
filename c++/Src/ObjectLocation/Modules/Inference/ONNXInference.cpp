/*******************************************************
 文件名：ONNXInference.cpp
 作者：
 描述：ONNX模型推理模块实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "ONNXInference.h"
#include "../../Factory/ModuleFactory.h"
#include <iostream>

// 注册模块
REGISTER_MODULE(ONNXInference, ONNXInference)

ONNXInference::ONNXInference()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference")
    , m_inputData(nullptr)
{
}

ONNXInference::~ONNXInference()
{
    m_session.release();
}

bool ONNXInference::init(CSelfAlgParam* p_pAlgParam)
{
    if (p_pAlgParam) {
        // 从配置参数中读取模型路径等参数
        // TODO: 实现参数读取逻辑
    }

    return loadModel();
}

void ONNXInference::setInput(void* input)
{
    m_inputData = input;
}

void* ONNXInference::getOutput()
{
    return m_outputData.data();
}

bool ONNXInference::loadModel()
{
    try {
        // 设置会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        if (m_params.device == "GPU") {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

        // 加载模型
        m_session = Ort::Session(m_env, m_params.modelPath.c_str(), session_options);

        // 获取输入输出节点名称
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 获取输入节点信息
        size_t num_input_nodes = m_session.GetInputCount();
        m_inputNames.reserve(num_input_nodes);
        m_inputNamePtrs.reserve(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = m_session.GetInputNameAllocated(i, allocator);
            m_inputNamePtrs.push_back(std::move(input_name));
            m_inputNames.push_back(m_inputNamePtrs.back().get());
        }

        // 获取输出节点信息
        size_t num_output_nodes = m_session.GetOutputCount();
        m_outputNames.reserve(num_output_nodes);
        m_outputNamePtrs.reserve(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = m_session.GetOutputNameAllocated(i, allocator);
            m_outputNamePtrs.push_back(std::move(output_name));
            m_outputNames.push_back(m_outputNamePtrs.back().get());
        }

        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

void* ONNXInference::execute()
{
    if (!m_inputData) {
        std::cerr << "Input data is null" << std::endl;
        return nullptr;
    }

    try {
        // 准备输入数据
        auto* input_tensor = static_cast<cv::Mat*>(m_inputData);
        std::vector<float> input_data;
        input_data.assign((float*)input_tensor->data, 
                         (float*)input_tensor->data + input_tensor->total() * input_tensor->channels());

        // 创建输入tensor
        std::vector<int64_t> input_shape = {1, 3, input_tensor->rows, input_tensor->cols};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // 运行推理
        auto output_tensors = m_session.Run(
            Ort::RunOptions{nullptr}, 
            m_inputNames.data(), 
            &input_tensor_ort, 
            1, 
            m_outputNames.data(), 
            m_outputNames.size());

        // 处理输出
        if (output_tensors.size() > 0) {
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            
            m_outputData.assign(output_data, output_data + output_size);
            return m_outputData.data();
        }

        return nullptr;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return nullptr;
    }
} 