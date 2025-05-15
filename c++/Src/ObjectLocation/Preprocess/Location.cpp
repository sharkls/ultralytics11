/*******************************************************
 文件名：Location.cpp
 作者：sharkls
 描述：目标定位预处理模块实现
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "Location.h"

// 注册模块
REGISTER_MODULE("ObjectLocation", Location, Location)

Location::~Location()
{
}

bool Location::init(void* p_pAlgParam)
{
    LOG(INFO) << "Location::init status: start ";
    // 1. 从配置参数中读取预处理参数
    if (!p_pAlgParam) {
        return false;
    }
    // 2. 参数格式转换
    objectlocation::TaskConfig* taskConfig = static_cast<objectlocation::TaskConfig*>(p_pAlgParam);
    iou_thres_ = taskConfig->iou_thres();
    num_keys_ = taskConfig->num_keys();
    bucket_size_ = taskConfig->bucket_size();
    max_distance_ = taskConfig->max_distance();
    m_config = *taskConfig; 
    LOG(INFO) << "Location::init status: success ";
    return true;
}

void Location::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputdata = *static_cast<CAlgResult*>(input);
}

void* Location::getOutput()
{
    return &m_outputdata;
}

void Location::execute()
{   

    LOG(INFO) << "Location::execute status: start ";
    if (m_inputdata.vecFrameResult().empty()) {
        LOG(ERROR) << "Input data is empty";
        return;
    }

    try {
        // 1. 获取多模态感知结果和姿态估计结果
        const auto& multiModalResult = m_inputdata.vecFrameResult()[0];
        const auto& poseResult = m_inputdata.vecFrameResult()[1];

        // 2. 获取深度信息
        const auto& depth_map = multiModalResult.tCameraSupplement().vecDistanceInfo();
        int depth_width = multiModalResult.tCameraSupplement().usWidth();
        int depth_height = multiModalResult.tCameraSupplement().usHeight();

        const auto& det_objs = multiModalResult.vecObjectResult();
        const auto& pose_objs = poseResult.vecObjectResult();

        std::vector<bool> pose_matched(pose_objs.size(), false);
        std::vector<bool> det_matched(det_objs.size(), false);

        CFrameResult outputResult;

        // 3. NMS匹配
        for (size_t i = 0; i < det_objs.size(); ++i) {
            const auto& det = det_objs[i];
            int best_j = -1;
            float best_iou = 0.0f;
            for (size_t j = 0; j < pose_objs.size(); ++j) {
                float iou = calc_iou(det, pose_objs[j]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_j = j;
                }
            }
            if (best_iou > iou_thres_ && best_j >= 0) {
                pose_matched[best_j] = true;
                det_matched[i] = true;
                // 4. 关键点深度统计
                const auto& keypoints = pose_objs[best_j].vecKeypoints();
                std::vector<float> kp_depths;
                for (const auto& kp : keypoints) 
                {
                    // 判断关键点是否有效
                    if (kp.x() != 0.0f || kp.y() != 0.0f || kp.confidence() != 0.0f) {
                        float x = kp.x();
                        float y = kp.y();
                        float d = get_depth(depth_map, depth_width, depth_height, x, y);
                        if (d > 0) kp_depths.push_back(d);
                    }
                }
                float final_depth = get_bucket_depth(kp_depths, bucket_size_);
                if (final_depth > max_distance_) final_depth = max_distance_;
                if (final_depth <= 0) final_depth = 0;
                // 5. 填入新目标
                CObjectResult obj = det;
                obj.fDistance(final_depth);
                outputResult.vecObjectResult().push_back(obj);
            }
        }

        // 6. 未匹配的det目标，取box中心点深度
        for (size_t i = 0; i < det_objs.size(); ++i) {
            if (!det_matched[i]) {
                const auto& det = det_objs[i];
                float cx = (det.fTopLeftX() + det.fBottomRightX()) / 2.0f;
                float cy = (det.fTopLeftY() + det.fBottomRightY()) / 2.0f;
                float d = get_depth(depth_map, depth_width, depth_height, cx, cy);
                if (d > max_distance_) d = max_distance_;
                if (d <= 0) d = 0;
                CObjectResult obj = det;
                obj.fDistance(d);
                outputResult.vecObjectResult().push_back(obj);
            }
        }

        // 7. 未匹配的pose目标（舍弃）

        // 8. 填入输出
        m_outputdata.vecFrameResult().clear();
        m_outputdata.vecFrameResult().push_back(outputResult);

        LOG(INFO) << "Location::execute status: success!";
        // if (status_) {
        //     save_bin(m_outputdata, "./Save_Data/objectlocation/result/processed_objectlocation_preprocess.bin"); // ObjectLocation/Preprocess
        // }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return;
    }
}

float Location::calc_iou(const CObjectResult& a, const CObjectResult& b) const 
{
    float x1 = std::max(a.fTopLeftX(), b.fTopLeftX());
    float y1 = std::max(a.fTopLeftY(), b.fTopLeftY());
    float x2 = std::min(a.fBottomRightX(), b.fBottomRightX());
    float y2 = std::min(a.fBottomRightY(), b.fBottomRightY());
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.fBottomRightX() - a.fTopLeftX()) * (a.fBottomRightY() - a.fTopLeftY());
    float area_b = (b.fBottomRightX() - b.fTopLeftX()) * (b.fBottomRightY() - b.fTopLeftY());
    float iou = inter / (area_a + area_b - inter + 1e-6f);
    return iou;
}

// 根据中心点获取深度
float Location::get_depth(const std::vector<float>& depth_map, int width, int height, float x, float y) const 
{
    // 越界保护
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0.0f;
    }
    int ix = static_cast<int>(std::round(x));
    int iy = static_cast<int>(std::round(y));
    return depth_map[iy * width + ix];
}

// 根据桶分布获取深度
float Location::get_bucket_depth(const std::vector<float>& depths, float bucket_size) const 
{
    if (depths.empty()) return 0.0f;
    std::map<int, std::vector<float>> buckets;
    for (float d : depths) {
        int bucket = static_cast<int>(d / bucket_size);
        buckets[bucket].push_back(d);
    }
    // 取最多的桶
    int max_bucket = -1, max_count = 0;
    for (const auto& kv : buckets) {
        if (kv.second.size() > max_count) {
            max_count = kv.second.size();
            max_bucket = kv.first;
        }
    }
    // 取该桶的中位数
    std::vector<float> vals = buckets[max_bucket];
    std::sort(vals.begin(), vals.end());
    return vals[vals.size() / 2];
}