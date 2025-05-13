#include "Preprocess.h"

/**
 * PreProcess构造函数
 * 功能：
 *   创建PreProcess对象时记录启动状态
 */
PreProcess::PreProcess()
{   
    LOG(INFO) << "PreProcess :: PreProcess  status:    Started." << std::endl;
}

/**
 * PreProcess析构函数
 * 功能：
 *   销毁PreProcess对象时执行的清理操作
 */
PreProcess::~PreProcess()
{
}

/**
 * 初始化多模态融合预处理部分参数
 * 参数：
 *   p_pAlgParam - 触发算法参数
 * 功能：
 *   初始化PreProcess对象的算法参数，包括设置输入图像尺寸
 */
void PreProcess::init(CSelfAlgParam* p_pAlgParam) 
{   
    LOG(INFO) << "PreProcess :: init status : start." << std::endl;
    if (!p_pAlgParam) {
        LOG(ERROR) << "The Preprocess InitAlgorithm incoming parameter is empty" << std::endl;
        return;
    }

    // 从参数中读取图像尺寸
    m_nInputHeight = p_pAlgParam->m_nResizeOutputHeight;
    m_nInputWidth = p_pAlgParam->m_nResizeOutputWidth;

    // 检查图像尺寸是否有效
    if (m_nInputHeight <= 0 || m_nInputWidth <= 0) {
        LOG(ERROR) << "Invalid input size: " << m_nInputWidth << "x" << m_nInputHeight << std::endl;
        return;
    }

    LOG(INFO) << "PreProcess :: init status : finish! Input size: " 
              << m_nInputWidth << "x" << m_nInputHeight << std::endl;
}   

/**
 * 预处理多模态数据
 * 参数：
 *   p_rgbImg - RGB图像
 *   p_irImg - 红外图像
 *   p_homographyMatrix - 单应性矩阵
 * 返回值：
 *   std::tuple<std::vector<float>, std::vector<float>, std::array<float, 9>> - 预处理后的RGB数据、IR数据和单应性矩阵
 */
std::tuple<std::vector<float>, std::vector<float>, std::array<float, 9>> 
PreProcess::preprocessMultiModalData(const cv::Mat& p_rgbImg, 
                                   const cv::Mat& p_irImg,
                                   const std::array<float, 9>& p_homographyMatrix)
{
    // 检查输入尺寸是否已初始化
    if (m_nInputHeight <= 0 || m_nInputWidth <= 0) {
        LOG(ERROR) << "Input size not initialized" << std::endl;
        return std::make_tuple(std::vector<float>(), std::vector<float>(), p_homographyMatrix);
    }

    // 计算letterbox参数
    float rRgb = std::min(static_cast<float>(m_nInputHeight) / p_rgbImg.rows,
                         static_cast<float>(m_nInputWidth) / p_rgbImg.cols);
    float rIr = std::min(static_cast<float>(m_nInputHeight) / p_irImg.rows,
                        static_cast<float>(m_nInputWidth) / p_irImg.cols);

    int newUnpadRgbW = static_cast<int>(p_rgbImg.cols * rRgb);
    int newUnpadRgbH = static_cast<int>(p_rgbImg.rows * rRgb);
    int newUnpadIrW = static_cast<int>(p_irImg.cols * rIr);
    int newUnpadIrH = static_cast<int>(p_irImg.rows * rIr);

    float dwRgb = (m_nInputWidth - newUnpadRgbW) / 2.0f;
    float dhRgb = (m_nInputHeight - newUnpadRgbH) / 2.0f;
    float dwIr = (m_nInputWidth - newUnpadIrW) / 2.0f;
    float dhIr = (m_nInputHeight - newUnpadIrH) / 2.0f;

    // 更新单应性矩阵
    std::vector<float> sRgb = {rRgb, 0, 0, 0, rRgb, 0, 0, 0, 1};
    std::vector<float> sIr = {rIr, 0, 0, 0, rIr, 0, 0, 0, 1};
    std::vector<float> tRgb = {1, 0, dwRgb, 0, 1, dhRgb, 0, 0, 1};
    std::vector<float> tIr = {1, 0, dwIr, 0, 1, dhIr, 0, 0, 1};

    // 预处理RGB图像
    cv::Mat rgbResized;
    cv::resize(p_rgbImg, rgbResized, cv::Size(newUnpadRgbW, newUnpadRgbH));
    cv::Mat rgbPadded(m_nInputHeight, m_nInputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    rgbResized.copyTo(rgbPadded(cv::Rect(dwRgb, dhRgb, newUnpadRgbW, newUnpadRgbH)));
    
    // 预处理IR图像
    cv::Mat irResized;
    cv::resize(p_irImg, irResized, cv::Size(newUnpadIrW, newUnpadIrH));
    cv::Mat irPadded(m_nInputHeight, m_nInputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    irResized.copyTo(irPadded(cv::Rect(dwIr, dhIr, newUnpadIrW, newUnpadIrH)));

    // 转换为float并归一化
    rgbPadded.convertTo(rgbPadded, CV_32FC3, 1.0/255.0);
    irPadded.convertTo(irPadded, CV_32FC3, 1.0/255.0);

    // HWC to CHW
    std::vector<float> rgbInput(3 * m_nInputHeight * m_nInputWidth);
    std::vector<float> irInput(3 * m_nInputHeight * m_nInputWidth);
    
    std::vector<cv::Mat> rgbChannels(3);
    std::vector<cv::Mat> irChannels(3);
    cv::split(rgbPadded, rgbChannels);
    cv::split(irPadded, irChannels);

    for (int c = 0; c < 3; ++c) {
        memcpy(rgbInput.data() + c * m_nInputHeight * m_nInputWidth,
               rgbChannels[c].data,
               m_nInputHeight * m_nInputWidth * sizeof(float));
        memcpy(irInput.data() + c * m_nInputHeight * m_nInputWidth,
               irChannels[c].data,
               m_nInputHeight * m_nInputWidth * sizeof(float));
    }

    return std::make_tuple(rgbInput, irInput, p_homographyMatrix);
}

/**
 * 执行多模态融合预处理算法
 * 功能：
 *   获取输入数据，执行预处理逻辑，并设置输出数据
 */
void PreProcess::execute()
{
    LOG(INFO) << "PreProcess::execute status: start." << std::endl;

    // 检查输入尺寸是否已初始化
    if (m_nInputHeight <= 0 || m_nInputWidth <= 0) {
        LOG(ERROR) << "Input size not initialized, please call init first" << std::endl;
        return;
    }

    // 获取输入数据
    CMultiModalSrcData multiModalSrcData = m_pCommonData->m_multiModalSrcData;
    std::vector<CVideoSrcData> multiModalData = multiModalSrcData.vecVideoSrcData();
    std::array<float, 9> homographyMatrix = multiModalSrcData.fHomographyMatrix();

    // 检查输入数据
    if (multiModalData.size() != 2) {
        LOG(ERROR) << "Invalid video data size: " << multiModalData.size() << std::endl;
        return;
    }

    // 获取RGB和IR图像数据
    CVideoSrcData rgbData = multiModalData[0];
    CVideoSrcData irData = multiModalData[1];

    // 转换为OpenCV Mat格式
    cv::Mat rgbImg(rgbData.usBmpLength(), rgbData.usBmpWidth(), CV_8UC3, rgbData.vecImageBuf().data());
    cv::Mat irImg(irData.usBmpLength(), irData.usBmpWidth(), CV_8UC3, irData.vecImageBuf().data());

    // 计算letterbox参数
    float rRgb = std::min(static_cast<float>(m_nInputHeight) / rgbImg.rows,
                         static_cast<float>(m_nInputWidth) / rgbImg.cols);
    float rIr = std::min(static_cast<float>(m_nInputHeight) / irImg.rows,
                        static_cast<float>(m_nInputWidth) / irImg.cols);

    int newUnpadRgbW = static_cast<int>(rgbImg.cols * rRgb);
    int newUnpadRgbH = static_cast<int>(rgbImg.rows * rRgb);
    int newUnpadIrW = static_cast<int>(irImg.cols * rIr);
    int newUnpadIrH = static_cast<int>(irImg.rows * rIr);

    float dwRgb = (m_nInputWidth - newUnpadRgbW) / 2.0f;
    float dhRgb = (m_nInputHeight - newUnpadRgbH) / 2.0f;
    float dwIr = (m_nInputWidth - newUnpadIrW) / 2.0f;
    float dhIr = (m_nInputHeight - newUnpadIrH) / 2.0f;

    // 保存letterbox信息
    m_pCommonData->preprocessedData.dw = dwRgb;
    m_pCommonData->preprocessedData.dh = dhRgb;
    m_pCommonData->preprocessedData.ratio = rRgb;

    // 执行预处理
    auto [rgbInput, irInput, processedHomography] = preprocessMultiModalData(rgbImg, irImg, homographyMatrix);

    // 检查预处理结果
    if (rgbInput.empty() || irInput.empty()) {
        LOG(ERROR) << "Preprocessing failed" << std::endl;
        return;
    }

    // 直接存储预处理结果到ICommonData中
    m_pCommonData->preprocessedData.rgbInput = std::move(rgbInput);
    m_pCommonData->preprocessedData.irInput = std::move(irInput);
    m_pCommonData->preprocessedData.homography = processedHomography;
    m_pCommonData->preprocessedData.inputHeight = m_nInputHeight;
    m_pCommonData->preprocessedData.inputWidth = m_nInputWidth;
    LOG(INFO) << "PreProcess::execute status : finish! " << std::endl;
}