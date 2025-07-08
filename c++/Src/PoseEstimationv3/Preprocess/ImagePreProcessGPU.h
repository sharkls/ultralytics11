/*******************************************************
 文件名：ImagePreProcessGPU.h
 作者：
 描述：GPU加速的图像预处理模块
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#ifndef IMAGE_PRE_PROCESS_GPU_H
#define IMAGE_PRE_PROCESS_GPU_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "log.h"
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "CAlgResult.h"
#include "PoseEstimation_conf.pb.h"
#include "../CPoseEstimationAlg.h"

// CUDA核函数外部声明（在.cu文件中实现）
extern "C" {
    void launchResizeKernel(uchar3* src, uchar3* dst, int src_width, int src_height, 
                           int dst_width, int dst_height, cudaStream_t stream);
    void launchNormalizeKernel(uchar3* src, float* dst, int width, int height, 
                              float scale, cudaStream_t stream);
    void launchHWCtoCHWKernel(float* src, float* dst, int width, int height, 
                             cudaStream_t stream);
    void launchPadImageKernel(float* src, float* dst, int src_width, int src_height, 
                             int dst_width, int dst_height, int pad_top, int pad_left, 
                             float pad_value, cudaStream_t stream);
    void launchBgrToRgbKernel(uchar3* bgr, uchar3* rgb, int width, int height, 
                             cudaStream_t stream);
    void launchBatchPreprocessKernel(uchar3* src_images, float* dst_images,
                                    int* src_widths, int* src_heights,
                                    int* dst_widths, int* dst_heights,
                                    int* target_widths, int* target_heights,
                                    int* pad_tops, int* pad_lefts,
                                    int batch_size, int max_src_width, int max_src_height,
                                    int max_target_width, int max_target_height,
                                    cudaStream_t stream);
}

class ImagePreProcessGPU : public IBaseModule {
public:
    ImagePreProcessGPU(const std::string& exe_path);
    ~ImagePreProcessGPU() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ImagePreProcessGPU"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // GPU预处理相关函数
    bool initCUDA();
    void cleanupCUDA();
    
    // GPU内存管理
    bool allocateGPUMemory(size_t max_image_size);
    void freeGPUMemory();
    
    // GPU预处理核心函数
    std::vector<float> processSingleImageGPU(const cv::Mat& srcImage, int& outWidth, int& outHeight);
    std::vector<float> processSingleImageGPUWithPadding(const cv::Mat& srcImage, int targetWidth, int targetHeight, 
                                                       float& ratio, int& padTop, int& padLeft);
    std::vector<std::vector<float>> processBatchImagesGPU(const std::vector<cv::Mat>& srcImages, int targetWidth, int targetHeight,
                                                          std::vector<float>& ratios, std::vector<int>& padTops, std::vector<int>& padLefts);
    
    // 新增：GPU内存直接处理函数
    bool processBatchImagesGPUInPlace(const std::vector<cv::Mat>& srcImages, int targetWidth, int targetHeight,
                                     MultiImagePreprocessResultGPU& gpuResult);
    bool processSingleImageGPUInPlace(const cv::Mat& srcImage, int targetWidth, int targetHeight,
                                     float* gpu_dst, size_t dst_offset, float& ratio, int& padTop, int& padLeft);
    
    bool uploadImageToGPU(const cv::Mat& image, void* gpu_buffer);
    bool downloadImageFromGPU(void* gpu_buffer, std::vector<float>& output, int width, int height);
    
    // GPU内核函数调用（调用外部C函数）
    void callResizeKernel(void* src, void* dst, int src_width, int src_height, 
                         int dst_width, int dst_height, cudaStream_t stream);
    void callNormalizeKernel(void* src, void* dst, int width, int height, 
                            float scale, cudaStream_t stream);
    void callHWCtoCHWKernel(void* src, void* dst, int width, int height, 
                           cudaStream_t stream);
    void callPadImageKernel(void* src, void* dst, int src_width, int src_height, 
                           int dst_width, int dst_height, int pad_top, int pad_left, 
                           float pad_value, cudaStream_t stream);
    void callBgrToRgbKernel(void* src, void* dst, int width, int height, 
                           cudaStream_t stream);

    posetimation::YOLOModelConfig m_poseConfig;            // 姿态估计任务配置参数
    CAlgResult m_inputData;                                // 输入数据（CAlgResult格式）
    MultiImagePreprocessResult m_outputResult;             // 多图像预处理结果（CPU版本，兼容性）
    MultiImagePreprocessResultGPU m_outputResultGPU;       // 多图像预处理结果（GPU版本，新版本）

    // 图像相关参数
    int max_model_size_;  // 模型输入最大尺寸
    int stride_;          // 模型最大步长
    int channels_;        // 图像通道数

    // CUDA相关成员
    cudaStream_t m_cudaStream;
    cublasHandle_t m_cublasHandle;
    void* m_gpuInputBuffer;      // GPU输入缓冲区
    void* m_gpuOutputBuffer;     // GPU输出缓冲区
    void* m_gpuTempBuffer;       // GPU临时缓冲区
    size_t m_maxGPUBufferSize;   // 最大GPU缓冲区大小
    bool m_cudaInitialized;      // CUDA初始化状态

    // 运行状态
    bool status_ = false;
    
    // 性能监控
    std::chrono::high_resolution_clock::time_point m_start_time;
};

#endif // IMAGE_PRE_PROCESS_GPU_H 