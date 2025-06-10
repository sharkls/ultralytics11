/*******************************************************
 文件名：Location.cpp
 作者：sharkls
 描述：目标定位预处理模块实现
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "Location.h"
#include "GlobalContext.h"

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

    // LOG(INFO) << "Location::execute status: start ";
    if (m_inputdata.vecFrameResult().empty()) {
        LOG(ERROR) << "Input data is empty";
        return;
    }

    try {
        CFrameResult outputResult;

        if (m_inputdata.vecFrameResult().size() == 1) {
            const auto& result = m_inputdata.vecFrameResult()[0];
            // 判断类型，直接填充到输出
            if (result.eDataType() == DATA_TYPE_POSEALG_RESULT || result.eDataType() == DATA_TYPE_MMALG_RESULT) {
                m_outputdata.vecFrameResult().clear();
                m_outputdata.vecFrameResult().push_back(result);
                LOG(INFO) << "Location::execute: only one result, directly output.";
                return;
            } else {
                LOG(ERROR) << "Unknown data type in vecFrameResult[0]";
                return;
            }
        } else if (m_inputdata.vecFrameResult().size() >= 2) {
            // 1. 获取多模态感知结果和姿态估计结果
            const auto& multiModalResult = m_inputdata.vecFrameResult()[0];
            const auto& poseResult = m_inputdata.vecFrameResult()[1];

            // 2. 获取深度信息
            const auto& depth_map = multiModalResult.tCameraSupplement().vecDistanceInfo();
            int depth_width = multiModalResult.tCameraSupplement().usWidth();
            int depth_height = multiModalResult.tCameraSupplement().usHeight();

            // std::cout << "depth_width :" << depth_width << ", depth_height :" << depth_height << std::endl;

            const auto& det_objs = multiModalResult.vecObjectResult();
            const auto& pose_objs = poseResult.vecObjectResult();

            std::vector<bool> pose_matched(pose_objs.size(), false);
            std::vector<bool> det_matched(det_objs.size(), false);

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
                    // 4. 使用姿态估计的深度计算逻辑
                    const auto& keypoints = pose_objs[best_j].vecKeypoints();
                    std::vector<float> allKeypointDepths;
                    
                    for (const auto& kp : keypoints) {
                        // 获取关键点坐标
                        float kx = kp.x();
                        float ky = kp.y();
                        
                        // 像素坐标转整数下标
                        int kix = static_cast<int>(kx + 0.5f);
                        int kiy = static_cast<int>(ky + 0.5f);
                        
                        // std::cout << "\n关键点(" << kx << "," << ky << ") 5×5区域深度值：" << std::endl;
                        
                        // 检查边界
                        if (kix >= 0 && kix < depth_width && kiy >= 0 && kiy < depth_height) {
                            // 遍历关键点周围 5×5 的区域
                            for (int dy = -2; dy <= 2; ++dy) {
                                for (int dx = -2; dx <= 2; ++dx) {
                                    int currentIx = kix + dx;
                                    int currentIy = kiy + dy;
                                    
                                    // 检查当前坐标是否在深度图范围内
                                    if (currentIx >= 0 && currentIx < depth_width && currentIy >= 0 && currentIy < depth_height) {
                                        int idx = currentIy * depth_width + currentIx;
                                        if (idx >= 0 && idx < depth_map.size()) {
                                            float depth = depth_map[idx];
                                            if (depth > 0.0f) {  // 只收集有效的深度值
                                                allKeypointDepths.push_back(depth);
                                            }
                                            // std::cout << std::fixed << std::setprecision(2) << depth << "\t";
                                        }
                                    } else {
                                        std::cout << "N/A\t";
                                    }
                                }
                                // std::cout << std::endl;
                            }
                        }
                        // std::cout << std::endl;
                    }
                    
                    float final_depth = 0.0f;
                    // 如果收集到足够的深度值
                    if (!allKeypointDepths.empty()) {
                        // 对深度值进行排序
                        std::sort(allKeypointDepths.begin(), allKeypointDepths.end());
                        
                        // 计算四分位数
                        size_t n = allKeypointDepths.size();
                        float q1 = allKeypointDepths[n * 0.25];
                        float q3 = allKeypointDepths[n * 0.75];
                        float iqr = q3 - q1;
                        
                        // 定义异常值的界限
                        float lower_bound = q1 - 1.5 * iqr;
                        float upper_bound = q3 + 1.5 * iqr;
                        
                        // 过滤掉异常值
                        std::vector<float> filtered_depths;
                        for (float depth : allKeypointDepths) {
                            if (depth >= lower_bound && depth <= upper_bound) {
                                filtered_depths.push_back(depth);
                            }
                        }
                        
                        if (!filtered_depths.empty()) {
                            // 计算中位数作为最终深度值
                            size_t mid = filtered_depths.size() / 2;
                            if (filtered_depths.size() % 2 == 0) {
                                final_depth = (filtered_depths[mid - 1] + filtered_depths[mid]) / 2.0f;
                            } else {
                                final_depth = filtered_depths[mid];
                            }
                        }
                    }

                    if (final_depth > max_distance_) final_depth = max_distance_;
                    if (final_depth <= 0) final_depth = 0;

                    // 5. 填入新目标
                    CObjectResult obj = det;
                    obj.fDistance(final_depth);
                    // 更新类别
                    obj.strClass(update_class(pose_objs[best_j].strClass(), det.strClass()));
                    outputResult.vecObjectResult().push_back(obj);
                }
            }

            // 6. 未匹配的det目标，使用目标检测的深度计算逻辑
            for (size_t i = 0; i < det_objs.size(); ++i) {
                if (!det_matched[i]) {
                    const auto& det = det_objs[i];
                    float cx = (det.fTopLeftX() + det.fBottomRightX()) / 2.0f;
                    float cy = (det.fTopLeftY() + det.fBottomRightY()) / 2.0f;
                    
                    // 像素坐标转整数下标
                    int ix = static_cast<int>(cx + 0.5f);
                    int iy = static_cast<int>(cy + 0.5f);
                    
                    // std::cout << "\n目标中心点(" << cx << "," << cy << ") 5×5区域深度值：" << std::endl;
                    
                    float final_depth = 0.0f;
                    // 检查边界
                    if (ix >= 0 && ix < depth_width && iy >= 0 && iy < depth_height) {
                        // 定义一个数组来存储 5×5 区域的深度值
                        std::vector<float> depthValues;
                        
                        // 遍历中心点周围 5×5 的区域
                        for (int dy = -2; dy <= 2; ++dy) {
                            for (int dx = -2; dx <= 2; ++dx) {
                                int currentIx = ix + dx;
                                int currentIy = iy + dy;
                                
                                // 检查当前坐标是否在深度图范围内
                                if (currentIx >= 0 && currentIx < depth_width && currentIy >= 0 && currentIy < depth_height) {
                                    int idx = currentIy * depth_width + currentIx;
                                    if (idx >= 0 && idx < depth_map.size()) {
                                        float depth = depth_map[idx];
                                        depthValues.push_back(depth);
                                        // std::cout << std::fixed << std::setprecision(2) << depth << "\t";
                                    }
                                } 
                                // else {
                                //     std::cout << "N/A\t";
                                // }
                            }
                            // std::cout << std::endl;
                        }
                        // std::cout << std::endl;
                        
                        // 如果收集到足够的深度值
                        if (!depthValues.empty()) {
                            // 去除深度值为 0 的点
                            std::vector<float> nonZeroDepthValues;
                            for (float depth : depthValues) {
                                if (depth != 0.0f) {
                                    nonZeroDepthValues.push_back(depth);
                                }
                            }
                            
                            // 如果存在非零深度值
                            if (!nonZeroDepthValues.empty()) {
                                // 如果非零深度值足够多（≥20 个），舍弃 10 个最小值和 10 个最大值
                                if (nonZeroDepthValues.size() >= 20) {
                                    std::sort(nonZeroDepthValues.begin(), nonZeroDepthValues.end());
                                    
                                    int startIdx = 10;
                                    int endIdx = nonZeroDepthValues.size() - 10;
                                    
                                    float sum = 0.0f;
                                    for (int i = startIdx; i < endIdx; ++i) {
                                        sum += nonZeroDepthValues[i];
                                    }
                                    final_depth = sum / (endIdx - startIdx);
                                } else {
                                    // 如果非零深度值不足 20 个，直接求平均值
                                    float sum = 0.0f;
                                    for (float depth : nonZeroDepthValues) {
                                        sum += depth;
                                    }
                                    final_depth = sum / nonZeroDepthValues.size();
                                }
                            }
                        }
                    }
                    
                    if (final_depth > max_distance_) final_depth = max_distance_;
                    if (final_depth <= 0) final_depth = 0;
                    
                    CObjectResult obj = det;
                    obj.fDistance(final_depth);
                    outputResult.vecObjectResult().push_back(obj);
                }
            }

            // 7. 未匹配的pose目标（舍弃）

            // 8. 填入输出
            m_outputdata.vecFrameResult().clear();
            m_outputdata.vecFrameResult().push_back(outputResult);

            // LOG(INFO) << "Location::execute status: success!";
            // if (status_) {
            //     save_bin(m_outputdata, "./Save_Data/objectlocation/result/processed_objectlocation_preprocess.bin"); // ObjectLocation/Preprocess
            // }
        }
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
float Location::get_depth(const std::vector<short>& depth_map, int width, int height, float x, float y) const 
{
    // 越界保护
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0.0f;
    }
    int ix = static_cast<int>(std::round(x));
    int iy = static_cast<int>(std::round(y));
    return static_cast<float>(depth_map[iy * width + ix]);
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

std::string Location::update_class(const std::string& pose_class, const std::string& multi_class) const 
{
    std::cout << "update_class : " << pose_class << ", " << multi_class << std::endl;
    // 将字符串转换为整数进行比较
    int pose_cls = std::stoi(pose_class);
    int multi_cls = std::stoi(multi_class);

    // 根据规则更新类别
    if (pose_cls == 0) {
        // pose类别为0时
        if (multi_cls == 1) {
            return "1";  // 保持multi类别不变
        } else {
            return "0";  // multi类别为0或2时，更改为0
        }
    } else if (pose_cls == 1) {
        // pose类别为1时
        if (multi_cls == 0) {
            return "2";  // multi类别为0时，更改为2
        } else {
            return multi_class;  // 其他情况保持multi类别不变
        }
        return multi_class;
    }

    // 默认返回multi类别
    std::cout << "multi_class : " << pose_class << ", " << multi_class << std::endl;
    return multi_class;
}