#include "EFDEYolo11.h"
#include "log.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>

// 添加互斥锁用于线程安全
static std::mutex g_mutex;

EFDEYolo11::EFDEYolo11() : m_pStream(nullptr), m_pInputBuffer(nullptr), m_pOutputBuffer(nullptr)
{
    LOG(INFO) << "EFDEYolo11 :: EFDEYolo11 status: Started." << std::endl;
}

EFDEYolo11::EFDEYolo11(const CSelfAlgParam* p_pAlgParam) : m_pStream(nullptr), m_pInputBuffer(nullptr), m_pOutputBuffer(nullptr)
{
    LOG(INFO) << "EFDEYolo11 :: EFDEYolo11 with param status: Started." << std::endl;
    init(const_cast<CSelfAlgParam*>(p_pAlgParam));
}

EFDEYolo11::~EFDEYolo11()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    
    // 清理CUDA资源
    if (m_pStream) {
        cudaStreamDestroy(m_pStream);
        m_pStream = nullptr;
    }
    if (m_pInputBuffer) {
        cudaFree(m_pInputBuffer);
        m_pInputBuffer = nullptr;
    }
    if (m_pOutputBuffer) {
        cudaFree(m_pOutputBuffer);
        m_pOutputBuffer = nullptr;
    }
}

void EFDEYolo11::init(CSelfAlgParam* p_pAlgParam)
{
    LOG(INFO) << "EFDEYolo11 :: init status: start." << std::endl;
    if (!p_pAlgParam) {
        LOG(ERROR) << "The EFDEYolo11 InitAlgorithm incoming parameter is empty" << std::endl;
        return;
    }

    // 从参数中读取配置
    m_strModelPath = p_pAlgParam->m_strEnginepath;
    m_nBatchSize = p_pAlgParam->m_nBatchSize;
    m_nInputChannels = p_pAlgParam->m_nInputChannels;
    m_nInputHeight = p_pAlgParam->m_nResizeOutputHeight;
    m_nInputWidth = p_pAlgParam->m_nResizeOutputWidth;

    // // 打印输入参数
    // LOG(INFO) << "Input parameters - Batch: " << m_nBatchSize 
    //           << ", Channels: " << m_nInputChannels 
    //           << ", Height: " << m_nInputHeight 
    //           << ", Width: " << m_nInputWidth << std::endl;

    // 设置NMS参数
    m_fConfThres = p_pAlgParam->m_fConfidenceThreshold;
    m_fIouThres = p_pAlgParam->m_fNmsThreshold;
    m_nNumClasses = p_pAlgParam->m_nNumClasses;
    m_nNumAnchors = p_pAlgParam->m_nNumAnchors;
    LOG(INFO) << "EFDEYolo11 :: init status: m_nNumAnchors: " << m_nNumAnchors << std::endl;

    // 初始化CUDA流
    cudaError_t cudaStatus = cudaStreamCreate(&m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to create CUDA stream: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // 加载engine文件
    std::ifstream file(m_strModelPath, std::ios::binary);
    if (!file) {
        LOG(ERROR) << "无法打开engine文件: " << m_strModelPath << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // 创建TensorRT运行时和引擎
    m_pRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if (!m_pRuntime) {
        LOG(ERROR) << "Failed to create TensorRT runtime" << std::endl;
        return;
    }

    m_pEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        m_pRuntime->deserializeCudaEngine(engineData.data(), size));
    if (!m_pEngine) {
        LOG(ERROR) << "Failed to deserialize TensorRT engine" << std::endl;
        return;
    }

    m_pContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        m_pEngine->createExecutionContext());
    if (!m_pContext) {
        LOG(ERROR) << "Failed to create TensorRT execution context" << std::endl;
        return;
    }

    // 获取输入输出信息
    int numBindings = m_pEngine->getNbIOTensors();
    for (int i = 0; i < numBindings; ++i) {
        auto dims = m_pEngine->getTensorShape(m_pEngine->getIOTensorName(i));
        if (m_pEngine->getTensorIOMode(m_pEngine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT) {
            m_vecInputDims.push_back(dims);
        } else {
            m_vecOutputDims.push_back(dims);
        }
    }

    // 计算输入数据的总大小（考虑RGB、IR和单应性矩阵）
    size_t imageSize = m_nBatchSize * m_nInputChannels * m_nInputHeight * m_nInputWidth * sizeof(float);
    size_t extrinsicsSize = m_nBatchSize * 9 * sizeof(float);
    size_t totalInputSize = imageSize + extrinsicsSize;

    // 打印调试信息
    // LOG(INFO) << "Allocating GPU memory - Image size: " << imageSize
    //           << ", Extrinsics size: " << extrinsicsSize
    //           << ", Total size: " << totalInputSize << " bytes" << std::endl;

    // 分配GPU内存（考虑内存对齐）
    cudaStatus = cudaMalloc(&m_pInputBuffer, totalInputSize);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate input buffer: " << cudaGetErrorString(cudaStatus) 
                  << " (requested size: " << totalInputSize << " bytes)" << std::endl;
        return;
    }

    // 验证分配的内存大小
    size_t freeMemory, totalMemory;
    cudaStatus = cudaMemGetInfo(&freeMemory, &totalMemory);
    if (cudaStatus == cudaSuccess) {
        LOG(INFO) << "Successfully allocated " << totalInputSize << " bytes on GPU"
                  << " (Free memory: " << freeMemory << ", Total memory: " << totalMemory << ")" << std::endl;
    }

    // 计算输出大小
    size_t outputSize = 1;
    for (int i = 0; i < m_vecOutputDims[0].nbDims; ++i) {
        outputSize *= m_vecOutputDims[0].d[i];
    }
    outputSize *= sizeof(float);

    // 分配输出缓冲区（考虑内存对齐）
    cudaStatus = cudaMalloc(&m_pOutputBuffer, outputSize);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate output buffer: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // // 打印所有输入输出张量名
    // for (int i = 0; i < m_pEngine->getNbIOTensors(); ++i) {
    //     const char* name = m_pEngine->getIOTensorName(i);
    //     auto mode = m_pEngine->getTensorIOMode(name);
    //     LOG(INFO) << "Tensor name: " << name << ", mode: " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
    // }

    // 在init()中分配三块输入buffer
    cudaMalloc(&m_pInputBuffer_images0, imageSize);
    cudaMalloc(&m_pInputBuffer_images1, imageSize);
    cudaMalloc(&m_pInputBuffer_extrinsics, extrinsicsSize);

    LOG(INFO) << "EFDEYolo11 :: init status: finish!" << std::endl;
}

void EFDEYolo11::execute()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    LOG(INFO) << "EFDEYolo11::execute status: start." << std::endl;

    // 获取预处理后的数据
    const auto& preprocessedData = m_pCommonData->preprocessedData;
    if (preprocessedData.rgbInput.empty() || preprocessedData.irInput.empty() || preprocessedData.homography.empty()) {
        LOG(ERROR) << "Preprocessed data is empty" << std::endl;
        return;
    }

    // 检查输入数据范围
    float minRgb = *std::min_element(preprocessedData.rgbInput.begin(), preprocessedData.rgbInput.end());
    float maxRgb = *std::max_element(preprocessedData.rgbInput.begin(), preprocessedData.rgbInput.end());
    float minIr = *std::min_element(preprocessedData.irInput.begin(), preprocessedData.irInput.end());
    float maxIr = *std::max_element(preprocessedData.irInput.begin(), preprocessedData.irInput.end());

    // LOG(INFO) << "Input data ranges - RGB: [" << minRgb << ", " << maxRgb 
    //           << "], IR: [" << minIr << ", " << maxIr << "]" << std::endl;

    // // 打印输入张量形状
    // LOG(INFO) << "Input tensor shapes:" << std::endl;
    // for (int i = 0; i < m_vecInputDims.size(); ++i) {
    //     LOG(INFO) << "Input " << i << ": ";
    //     for (int j = 0; j < m_vecInputDims[i].nbDims; ++j) {
    //         LOG(INFO) << m_vecInputDims[i].d[j] << " ";
    //     }
    //     LOG(INFO) << std::endl;
    // }

    // // 打印输出张量形状
    // LOG(INFO) << "Output tensor shapes:" << std::endl;
    // for (int i = 0; i < m_vecOutputDims.size(); ++i) {
    //     LOG(INFO) << "Output " << i << ": ";
    //     for (int j = 0; j < m_vecOutputDims[i].nbDims; ++j) {
    //         LOG(INFO) << m_vecOutputDims[i].d[j] << " ";
    //     }
    //     LOG(INFO) << std::endl;
    // }

    // // 添加letterbox信息调试
    // LOG(INFO) << "LetterBox info - dw: " << preprocessedData.dw 
    //           << ", dh: " << preprocessedData.dh 
    //           << ", ratio: " << preprocessedData.ratio << std::endl;

    // // 打印调试信息
    // LOG(INFO) << "RGB input size: " << preprocessedData.rgbInput.size() 
    //           << ", IR input size: " << preprocessedData.irInput.size()
    //           << ", Homography size: " << preprocessedData.homography.size() << std::endl;

    // 计算输入数据的总大小
    size_t imageSize = preprocessedData.rgbInput.size() * sizeof(float);
    size_t extrinsicsSize = preprocessedData.homography.size() * sizeof(float);
    size_t totalSize = imageSize + extrinsicsSize;
    // LOG(INFO) << "Total input size: " << totalSize << " bytes" << std::endl;

    // 使用 void* 进行内存拷贝，避免类型转换问题
    void* inputBuffer = m_pInputBuffer;

    // 拷贝RGB数据到GPU
    cudaError_t cudaStatus = cudaMemcpyAsync(m_pInputBuffer_images0, 
                   preprocessedData.rgbInput.data(),
                   imageSize, 
                   cudaMemcpyHostToDevice, 
                   m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy RGB data to GPU: " << cudaGetErrorString(cudaStatus) 
                  << " (size: " << imageSize << " bytes)" << std::endl;
        return;
    }

    // 拷贝IR数据到GPU
    cudaStatus = cudaMemcpyAsync(m_pInputBuffer_images1, 
                   preprocessedData.irInput.data(),
                   imageSize, 
                   cudaMemcpyHostToDevice, 
                   m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy IR data to GPU: " << cudaGetErrorString(cudaStatus) 
                  << " (size: " << imageSize << " bytes)" << std::endl;
        return;
    }

    // 拷贝单应性矩阵到GPU
    cudaStatus = cudaMemcpyAsync(m_pInputBuffer_extrinsics, 
                   preprocessedData.homography.data(),
                   extrinsicsSize, 
                   cudaMemcpyHostToDevice, 
                   m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy homography matrix to GPU: " << cudaGetErrorString(cudaStatus) 
                  << " (size: " << extrinsicsSize << " bytes)" << std::endl;
        return;
    }

    // 分别设置每个输入张量的地址
    m_pContext->setTensorAddress("images", m_pInputBuffer_images0);
    m_pContext->setTensorAddress("images2", m_pInputBuffer_images1);
    m_pContext->setTensorAddress("extrinsics", m_pInputBuffer_extrinsics);
    m_pContext->setTensorAddress("output0", m_pOutputBuffer);

    // // 打印张量信息
    // LOG(INFO) << "Input tensor shape: " << m_vecInputDims[0].d[0] << "x" 
    //           << m_vecInputDims[0].d[1] << "x" 
    //           << m_vecInputDims[0].d[2] << "x" 
    //           << m_vecInputDims[0].d[3] << std::endl;

    // 执行推理
    bool status = m_pContext->enqueueV3(m_pStream);
    if (!status) {
        LOG(ERROR) << "TensorRT inference failed" << std::endl;
        return;
    }

    // 同步CUDA流
    cudaStatus = cudaStreamSynchronize(m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to synchronize CUDA stream: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // 获取输出
    size_t outputSize = 1;
    for (int i = 0; i < m_vecOutputDims[0].nbDims; ++i) {
        outputSize *= m_vecOutputDims[0].d[i];
    }
    std::vector<float> output(outputSize);
    
    // 从GPU拷贝结果到主机
    cudaStatus = cudaMemcpyAsync(output.data(), m_pOutputBuffer,
                   outputSize * sizeof(float),
                   cudaMemcpyDeviceToHost, m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to copy results from GPU: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // 同步CUDA流
    cudaStatus = cudaStreamSynchronize(m_pStream);
    if (cudaStatus != cudaSuccess) {
        LOG(ERROR) << "Failed to synchronize CUDA stream: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // // 添加输出调试信息
    // LOG(INFO) << "Output size: " << output.size() << std::endl;
    // LOG(INFO) << "Output range: [" << *std::min_element(output.begin(), output.end()) 
    //           << ", " << *std::max_element(output.begin(), output.end()) << "]" << std::endl;

    // 处理输出结果
    LetterBoxInfo letterboxInfo;
    letterboxInfo.dw = preprocessedData.dw;
    letterboxInfo.dh = preprocessedData.dh;
    letterboxInfo.ratio = preprocessedData.ratio;

    // 处理输出并应用NMS
    auto results = processOutput(output, m_fConfThres, m_fIouThres, m_nNumClasses, letterboxInfo);
    
    // LOG(INFO) << "Raw detection results before NMS: " << results.size() << std::endl;
    // LOG(INFO) << "EFDEYolo11::execute results size: " << results.size() << std::endl;

    // 将结果转换为CAlgResult格式
    CAlgResult algResult;
    CFrameResult frameResult;
    for (const auto& result : results) {
        CObjectResult objectResult;
        objectResult.fTopLeftX(result[0]);
        objectResult.fTopLeftY(result[1]);
        objectResult.fBottomRightX(result[2]);
        objectResult.fBottomRightY(result[3]);
        objectResult.fVideoConfidence(result[4]);
        objectResult.strClass(std::to_string(static_cast<int>(result[5])));
        frameResult.vecObjectResult().push_back(objectResult);
        
    }
    algResult.vecFrameResult().push_back(frameResult);
    // 存储结果
    m_pCommonData->m_multiModalFusionAlgResult = algResult;
    LOG(INFO) << "EFDEYolo11::execute FrameNums: " << m_pCommonData->m_multiModalFusionAlgResult.vecFrameResult().size() << std::endl;
    if (m_pCommonData->m_multiModalFusionAlgResult.vecFrameResult().size() > 0) {
        LOG(INFO) << "EFDEYolo11::execute ObjectNums: " << m_pCommonData->m_multiModalFusionAlgResult.vecFrameResult()[0].vecObjectResult().size() << std::endl;
    }

    LOG(INFO) << "EFDEYolo11::execute status: finish!" << std::endl;
}

// 辅助函数：处理输出结果
std::vector<std::vector<float>> EFDEYolo11::processOutput(
    const std::vector<float>& output,
    float confThres,
    float iouThres,
    int numClasses,
    const LetterBoxInfo& letterboxInfo)
{
    // LOG(INFO) << "Processing output with conf_thres: " << confThres 
    //           << ", iou_thres: " << iouThres 
    //           << ", num_classes: " << numClasses << std::endl;
    
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    // // 打印输出张量形状
    // LOG(INFO) << "Output tensor shape: ";
    // for (int i = 0; i < m_vecOutputDims[0].nbDims; ++i) {
    //     LOG(INFO) << m_vecOutputDims[0].d[i] << " ";
    // }
    // LOG(INFO) << std::endl;

    // 解析输出
    for (int i = 0; i < m_nNumAnchors; ++i) {
        float maxConf = 0.0f;
        int maxClass = 0;
        
        // 获取类别分数
        for (int c = 0; c < numClasses; ++c) {
            // 修改索引计算方式，与trt_infer保持一致
            float conf = output[(4 + c) * m_nNumAnchors + i];
            if (conf > maxConf) {
                maxConf = conf;
                maxClass = c;
            }
        }

        if (maxConf > confThres) {
            // LOG(INFO) << "Found detection - conf: " << maxConf 
            //           << ", class: " << maxClass << std::endl;
            
            // 获取边界框坐标
            float x = output[i];  // output[0][i]
            float y = output[m_nNumAnchors + i];  // output[1][i]
            float w = output[2 * m_nNumAnchors + i];  // output[2][i]
            float h = output[3 * m_nNumAnchors + i];  // output[3][i]

            // // 打印原始坐标
            // LOG(INFO) << "Original coordinates - x: " << x 
            //           << ", y: " << y 
            //           << ", w: " << w 
            //           << ", h: " << h << std::endl;

            // 转换为xyxy格式
            float x1 = x - w/2;
            float y1 = y - h/2;
            float x2 = x + w/2;
            float y2 = y + h/2;

            // 应用letterbox变换
            x1 = (x1 - letterboxInfo.dw) / letterboxInfo.ratio;
            y1 = (y1 - letterboxInfo.dh) / letterboxInfo.ratio;
            x2 = (x2 - letterboxInfo.dw) / letterboxInfo.ratio;
            y2 = (y2 - letterboxInfo.dh) / letterboxInfo.ratio;

            // // 打印变换后的坐标
            // LOG(INFO) << "Transformed coordinates - x1: " << x1 
            //           << ", y1: " << y1 
            //           << ", x2: " << x2 
            //           << ", y2: " << y2 << std::endl;

            // 检查边界框是否有效
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1) {
                boxes.push_back({x1, y1, x2, y2});
                scores.push_back(maxConf);
                classIds.push_back(maxClass);
            }
        }
    }

    // 对每个类别分别应用NMS
    std::vector<std::vector<float>> results;
    for (int clsId = 0; clsId < numClasses; ++clsId) {
        std::vector<std::vector<float>> clsBoxes;
        std::vector<float> clsScores;
        std::vector<int> indices;
        
        // 收集当前类别的检测框
        for (size_t i = 0; i < classIds.size(); ++i) {
            if (classIds[i] == clsId) {
                clsBoxes.push_back(boxes[i]);
                clsScores.push_back(scores[i]);
                indices.push_back(i);
            }
        }
        
        if (clsBoxes.empty()) continue;
        
        // 应用NMS
        std::vector<int> keep = nms(clsBoxes, clsScores, iouThres);
        
        // 添加结果
        for (int i : keep) {
            results.push_back({clsBoxes[i][0], clsBoxes[i][1], clsBoxes[i][2], clsBoxes[i][3],
                             clsScores[i], static_cast<float>(clsId)});
        }
    }

    return results;
}

// 辅助函数：NMS实现
std::vector<int> EFDEYolo11::nms(const std::vector<std::vector<float>>& boxes,
                                const std::vector<float>& scores,
                                float iouThres)
{
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);
        indices.erase(indices.begin());

        std::vector<int> tmpIndices;
        for (int i : indices) {
            float iou = 0.0f;
            // 计算IOU
            float xx1 = std::max(boxes[idx][0], boxes[i][0]);
            float yy1 = std::max(boxes[idx][1], boxes[i][1]);
            float xx2 = std::min(boxes[idx][2], boxes[i][2]);
            float yy2 = std::min(boxes[idx][3], boxes[i][3]);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;

            float area1 = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]);
            float area2 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
            float unionArea = area1 + area2 - intersection;

            iou = intersection / (unionArea + 1e-16f);

            if (iou <= iouThres) {
                tmpIndices.push_back(i);
            }
        }
        indices = tmpIndices;
    }
    return keep;
}
