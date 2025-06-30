/*******************************************************
 文件名：ImagePreProcess.h
 作者：
 描述：图像预处理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef IMAGE_PRE_PROCESS_H
#define IMAGE_PRE_PROCESS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "log.h"
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "CAlgResult.h"
#include "PoseEstimation_conf.pb.h"
#include "../CPoseEstimationAlg.h"

// // CUDA预处理内核函数声明
// extern "C" {
//     // BGR到RGB颜色空间转换
//     void cuda_bgr_to_rgb(const unsigned char* bgr, float* rgb, int width, int height, cudaStream_t stream);
    
//     // 图像resize（双线性插值）
//     void cuda_resize_bilinear(const unsigned char* src, unsigned char* dst, 
//                               int src_width, int src_height, 
//                               int dst_width, int dst_height, 
//                               cudaStream_t stream);
    
//     // 图像归一化 (0-255 -> 0-1)
//     void cuda_normalize(const unsigned char* src, float* dst, int width, int height, cudaStream_t stream);
    
//     // HWC到CHW格式转换
//     void cuda_hwc_to_chw(const float* hwc, float* chw, int width, int height, int channels, cudaStream_t stream);
    
//     // 图像填充
//     void cuda_pad_image(const float* src, float* dst, 
//                        int src_width, int src_height, 
//                        int dst_width, int dst_height,
//                        int pad_top, int pad_left, 
//                        float pad_value, cudaStream_t stream);
    
//     // 批量预处理（组合多个操作）
//     void cuda_batch_preprocess(const unsigned char* bgr_images, float* processed_images,
//                               int* original_sizes, int* target_sizes, int* pad_params,
//                               int batch_size, int max_width, int max_height,
//                               cudaStream_t stream);
// }

class ImagePreProcess : public IBaseModule {
public:
    ImagePreProcess(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~ImagePreProcess() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ImagePreProcess"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // CPU处理单个子图的预处理
    std::vector<float> processSingleImage(const cv::Mat& srcImage, int& outWidth, int& outHeight);
    
    // CPU处理单个子图的预处理（带填充，用于批量处理）
    std::vector<float> processSingleImageWithPadding(const cv::Mat& srcImage, int targetWidth, int targetHeight, 
                                                     float& ratio, int& padTop, int& padLeft);

    // CUDA处理单个子图的预处理
    std::vector<float> processSingleImageCUDA(const cv::Mat& srcImage, int targetWidth, int targetHeight, 
                                             float& ratio, int& padTop, int& padLeft);
    
    // CUDA批量预处理
    void processBatchCUDA(const std::vector<cv::Mat>& srcImages, int targetWidth, int targetHeight);
    
    // 初始化CUDA资源
    bool initCUDA();
    
    // 清理CUDA资源
    void cleanupCUDA();

   posetimation::YOLOModelConfig m_poseConfig;            // 姿态估计任务配置参数
   CAlgResult m_inputData;                                // 输入数据（CAlgResult格式）
   MultiImagePreprocessResult m_outputResult;             // 多图像预处理结果

   // 图像相关参数
   int max_model_size_;  // 模型输入最大尺寸
   int stride_;          // 模型最大步长

   // 运行状态
   bool status_ = false;
   
   // CUDA相关成员
   bool m_useCUDA = true;                    // 是否使用CUDA加速
   cudaStream_t m_cudaStream = nullptr;      // CUDA流
   void* m_deviceBGR = nullptr;              // GPU上的BGR图像数据
   void* m_deviceRGB = nullptr;              // GPU上的RGB图像数据
   void* m_deviceProcessed = nullptr;        // GPU上的处理后数据
   void* m_deviceTemp = nullptr;             // GPU上的临时数据
   size_t m_maxImageSize = 0;                // 最大图像尺寸（用于内存分配）
   size_t m_maxBatchSize = 8;                // 最大批处理大小
};

#endif // IMAGE_PRE_PROCESS_H 