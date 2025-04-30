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

文件名：CCameraParam.idl

```
struct C485Param
{
    string          strUsbDev;      //USB串口设备文件路径
    unsigned long   unBaudRate;     //波特率
    float           fTimeout;       //超时时间
    unsigned long   unId;           //id,对应相机id
};
struct CCameraDev
{
    string                strCameraIp;            //相机IP
    octet                 ucCameraId;             //相机序号
    string                strCameraUser;          //相机用户名
    string                strCameraPwd;           //相机用户密码
    sequence<float>       vecInParameter;         //内参
    sequence<float>       vecRotateMatrix;        //旋转矩阵
    sequence<float>       vecTranslationMatrix;   //平移矩阵
    sequence<float>       vecDistMatrix;          //畸变系数
    C485Param             Camera485Param;         //485参数
};
struct CCameraParam
{
    octet                   unCameraCount;      //相机个数
    boolean                 bUse485;            //是否使用485
    sequence<CCameraDev>    vecCameraDev;       //相机设备参数
};
```

通过fastddsgen工具生成消息类

```bash
fastddsgen <idl文件名> -d <生成文件目录>
```


生成文件：CCameraParam.h、CCameraParam.cxx、CCameraParamPubSubTypes.h、CCameraParamPubSubTypes.cxx

* CCameraParam.h、CCameraParam.cxx：为定义的消息类
* CCameraParamPubSubTypes.h、CCameraParamPubSubTypes.cxx：为注册消息类使用的类
* CDataBase 的析构函数默认生成的不是 virtual 类型，需要自行添加，否则会出现内存释放不完全的问题
