#ifndef SGBM_H
#define SGBM_H

#include <opencv2/opencv.hpp>
#include <string>
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "CMultiModalSrcData.h"
#include "BinocularPositioning_conf.pb.h"

class SGBM : public IBaseModule {
public:
    SGBM(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~SGBM() override;

    // 实现基类接口
    std::string getModuleName() const override { return "SGBM"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

    // 设置SGBM参数
    bool setParameters(int minDisparity = 0,                 // 最小视差值
              int numDisparities = 128,             // 视差搜索范围（16倍数）
              int blockSize = 5,                    // 匹配块大小（奇数， 值越大匹配越稳定但是会丢失细节）
              int P1 = 8 * 3 * 5 * 5,               // 控制视差平滑度的第一个参数（值越大，视差图越平滑， blockSize * blockSize * 3 的倍数）
              int P2 = 32 * 3 * 5 * 5,              // 控制视差平滑度的第二个参数（值越大，视差图越平滑， P1 的 3-4 倍）
              int disp12MaxDiff = 1,                // 左右视差检查的最大允许差异
              int preFilterCap = 63,                // 预处理滤波器的截断值(值越大，预处理效果越强)
              int uniquenessRatio = 15,             // 唯一性比率 (用于过滤不可靠的匹配, 值越大，匹配要求越严格)
              int speckleWindowSize = 100,          // 斑点窗口大小(用于过滤小的视差区域, 值越大，过滤效果越强)
              int speckleRange = 2,                 // 斑点范围(用于过滤小的视差区域, 值越大，过滤效果越强)
              int mode = cv::StereoSGBM::MODE_SGBM);    // 模式（MODE_SGBM：标准SGBM算法 \ MODE_HH：全尺寸双通道算法 \ MODE_SGBM_3WAY）

    // 计算视差图
    cv::Mat computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg);

    // 计算深度图
    cv::Mat computeDepth(const cv::Mat& disparity);

    // 获取运行状态
    bool getStatus() const { return status_; }

    // 可视化并保存视差图
    void saveDisparityVis(const cv::Mat& disparity, const std::string& path, bool useColor = false);

private:
    // SGBM参数
    int minDisparity_;                 // 最小视差值
    int numDisparities_;              // 视差搜索范围（16倍数）
    int blockSize_;                   // 匹配块大小
    int P1_;                          // 控制视差平滑度的第一个参数
    int P2_;                          // 控制视差平滑度的第二个参数
    int disp12MaxDiff_;               // 左右视差检查的最大允许差异
    int preFilterCap_;                // 预处理滤波器的截断值
    int uniquenessRatio_;             // 唯一性比率
    int speckleWindowSize_;           // 斑点窗口大小
    int speckleRange_;                // 斑点范围
    int mode_;                        // 模式

    // 相机参数
    float baseline_;                  // 相机基线长度(mm)
    int focusPixel_;                  // 像素焦距，单位为像素
    float focalLength_;               // 相机焦距(mm)

    // 运行状态
    bool status_ = false;

    // 内部成员
    cv::Ptr<cv::StereoSGBM> sgbm;
    cv::Mat disparity;
    cv::Mat depth;
    CMultiModalSrcData m_inputImage;         // 输入数据
    CAlgResult m_outputResult;               // 输出结果
};

#endif // SGBM_H 