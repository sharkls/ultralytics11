#### 映射关系

1. 基础类型

IDL                           | C++
------------------------------|----------------------------------
char 	                        | char       
octet 	                      | uint8_t    
short 	                      | int16_t    
unsigned short	              | uint16_t   
long                          | int32_t    
unsigned long                 | uint32_t   
long long	                    | int64_t    
unsigned long long	          | uint64_t   
float                         | float      
double                        | double     
boolean 	                    | bool       
string 	                      | std::string

2. 数组类型

IDL                           | C++
------------------------------|----------------------------------
char a[5]                     | std::array<char,5> a
octet a[5]                    | std::array<uint8_t,5> a
short a[5]                    | std::array<int16_t,5> a
unsigned short a[5]           | std::array<uint16_t,5> a
long a[5]                     | std::array<int32_t,5> a
unsigned long a[5]            | std::array<uint32_t,5> a
long long a[5]                | std::array<int64_t,5> a
unsigned long long a[5]       | std::array<uint64_t,5> a
float a[5]                    | std::array<float,5> a
double a[5]                   | std::array<double,5> a

3. 序列

IDL                           | C++
------------------------------|----------------------------------
sequence<char>                | std::vector<char>
sequence<octet>               | std::vector<uint8_t>
sequence<short>               | std::vector<int16_t>
sequence<unsigned short>      | std::vector<uint16_t>
sequence<long>                | std::vector<int32_t>
sequence<unsigned long>       | std::vector<uint32_t>
sequence<long long>           | std::vector<int64_t>
sequence<unsigned long long>  | std::vector<uint64_t>
sequence<float>               | std::vector<float>
sequence<double>              | std::vector<double>

4. Maps

IDL                           | C++
------------------------------|----------------------------------
map<char, unsigned long long> | std::map<char, uint64_t>

5. 枚举

```c++
enum Enumeration {    
  RED,    
  GREEN,    
  BLUE 
};
```

6. 结构体

```c++
struct Structure
{    
  octet octet_value;
  long long_value;
  string string_value;
};
```

#### 示例

文件名：CRadarSrcData.idl
```
// 毫米波雷达目标点信息
struct CRadarPoint {
    float               fRange;          // 径向距离(m)
    float               fAzimuth;        // 方位角(度)
    float               fElevation;      // 俯仰角(度)
    float               fDopplerVel;     // 多普勒速度(m/s)
    float               fRCS;            // 雷达散射截面积(dBsm)
    float               fSNR;            // 信噪比(dB)
    octet               ucPowerLevel;    // 信号强度等级
};

// 毫米波雷达目标跟踪信息
struct CRadarTrack {
    unsigned short      usTrackId;       // 目标跟踪ID
    float               fPosX;           // X坐标(m)
    float               fPosY;           // Y坐标(m)
    float               fPosZ;           // Z坐标(m)
    float               fVelX;           // X方向速度(m/s)
    float               fVelY;           // Y方向速度(m/s)
    float               fVelZ;           // Z方向速度(m/s)
    float               fAccX;           // X方向加速度(m/s²)
    float               fAccY;           // Y方向加速度(m/s²)
    float               fAccZ;           // Z方向加速度(m/s²)
    float               fLength;         // 目标长度(m)
    float               fWidth;          // 目标宽度(m)
    float               fHeight;         // 目标高度(m)
    float               fOrientation;    // 目标朝向角(度)
    float               fConfidence;    // 跟踪置信度
    octet               ucClassification;// 目标分类
};

// 毫米波雷达状态信息
struct CRadarStatus {
    octet               ucRadarState;    // 雷达工作状态
    float               fTemperature;    // 雷达温度
    octet               ucBlockage;      // 雷达遮挡状态
    octet               ucAlignment;     // 雷达对准状态
};

// 时间匹配好的毫米波数据结构体
struct CRadarSrcDataTimematch : CDataBase {
    octet                        ucRadarId;           // 雷达ID
    CRadarStatus                 tRadarStatus;        // 雷达状态信息
    sequence<CRadarPoint>        vecPoints;           // 原始点云数据
    sequence<CRadarTrack>        vecTracks;           // 目标跟踪数据
    float                        fNoisePower;         // 噪声功率
    float                        fInterference;       // 干扰水平
};
```

通过fastddsgen工具生成消息类

```bash
fastddsgen <idl文件名> -d <生成文件目录>
```


生成文件：CRadarSrcData.h、CRadarSrcData.cxx、CRadarSrcDataPubSubTypes.h、CRadarSrcDataPubSubTypes.cxx

* CRadarSrcData.h、CRadarSrcData.cxx：为定义的消息类
* CRadarSrcDataPubSubTypes.h、CRadarSrcDataPubSubTypes.cxx：为注册消息类使用的类
* CDataBase 的析构函数默认生成的不是 virtual 类型，需要自行添加，否则会出现内存释放不完全的问题
