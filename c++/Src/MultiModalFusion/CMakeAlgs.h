#ifndef CMAKEALGS_MULTIMODALFUSION_H
#define CMAKEALGS_MULTIMODALFUSION_H

#include <iostream>
#include <memory>
#include "IMakeAlg.h"
#include "ICommonAlg.h"

// 多模态融合
#include "Preprocess.h"
#include "EFDEYolo11.h"
#include "Postprocess.h"

class CMakeMultiModalFusionAlg_strategy final : public IMakeAlg {
public:
    CMakeMultiModalFusionAlg_strategy();
    explicit CMakeMultiModalFusionAlg_strategy(std::string& p_strConfigPath) {   
        LOG(INFO) << p_strConfigPath << std::endl;
        // 读取配置文件中算法选择参数
        YAML::Node config = YAML::LoadFile(p_strConfigPath);
        m_bPreIsValid = config["MULTIMODAL_FUSION"]["PRE_ISVALID"].as<bool>();
        m_bInferenceIsValid = config["MULTIMODAL_FUSION"]["Inference_ISVALID"].as<bool>();
        m_bPostIsValid = config["MULTIMODAL_FUSION"]["POST_ISVALID"].as<bool>();
        m_strPreAlgorithmName = config["MULTIMODAL_FUSION"]["PRE_ALG"].as<std::string>();
        m_strInferenceAlgorithmName = config["MULTIMODAL_FUSION"]["Inference_ALG"].as<std::string>();
        m_strPostAlgorithmName = config["MULTIMODAL_FUSION"]["POST_ALG"].as<std::string>();
    }

    ~CMakeMultiModalFusionAlg_strategy() = default;

    void makePre() override {
        if(m_bPreIsValid) {   
            m_pPreAlgorithm = m_mapPreAlgorithms[m_strPreAlgorithmName];
            m_vecAlgorithms.push_back(m_pPreAlgorithm);
        }
    }

    void makeInf() override {
        if(m_bInferenceIsValid) {   
            m_pInferenceAlgorithm = m_mapInferenceAlgorithms[m_strInferenceAlgorithmName];
            m_vecAlgorithms.push_back(m_pInferenceAlgorithm);
        }
    }

    void makePost() override {
        if(m_bPostIsValid) {   
            m_pPostAlgorithm = m_mapPostAlgorithms[m_strPostAlgorithmName];
            m_vecAlgorithms.push_back(m_pPostAlgorithm);
        }
    }

public:
    void makeAlgs() override {
        makePre();
        makeInf();
        makePost();
    }

private:
    // 是否开启各模块算法
    bool m_bPreIsValid;
    bool m_bInferenceIsValid;
    bool m_bPostIsValid;

    // 各模块对应算法选择
    std::string m_strPreAlgorithmName;
    std::string m_strInferenceAlgorithmName;
    std::string m_strPostAlgorithmName;

    // 各模块中对应算法
    std::map<std::string, ICommonAlgPtr> m_mapPreAlgorithms = {{"pre_1", std::make_shared<PreProcess>()}};
    std::map<std::string, ICommonAlgPtr> m_mapInferenceAlgorithms = {{"EFDEYolo11", std::make_shared<EFDEYolo11>()}};
    std::map<std::string, ICommonAlgPtr> m_mapPostAlgorithms = {{"post_1", std::make_shared<PostProcess>()}};
};

#endif //CMAKEALGS_MULTIMODALFUSION_H