/*******************************************************
 文件名：ONNXInference.h
 作者：
 描述：ONNX模型推理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include "../../../Common/IBaseModule.h"
#include <onnxruntime_cxx_api.h>
#include <vector>

class ONNXInference : public IBaseModule {
public:
    ONNXInference();
    ~ONNXInference() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ONNXInference"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(CSelfAlgParam* p_pAlgParam) override;
    void* execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // 推理参数
    struct InferenceParams {
        std::string modelPath;
        std::string device = "CPU";  // CPU or GPU
        int batchSize = 1;
    };

    bool loadModel();
    void preprocessInput();
    void postprocessOutput();

    InferenceParams m_params;
    Ort::Session m_session{nullptr};
    Ort::Env m_env;
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<Ort::AllocatedStringPtr> m_inputNamePtrs;
    std::vector<Ort::AllocatedStringPtr> m_outputNamePtrs;
    
    void* m_inputData;
    std::vector<float> m_outputData;
};

#endif // ONNX_INFERENCE_H 