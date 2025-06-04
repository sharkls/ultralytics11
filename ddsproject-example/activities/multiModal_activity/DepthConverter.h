#include <cstdint>

#include <vector>
#include <array>
#include <algorithm>

class DepthConverter
{
public:
    DepthConverter(const int width, const int height): m_width(width), m_height(height) {}

    std::vector<int32_t> process(const std::vector<uint8_t>& input, std::vector<int32_t>& result)
    {
        // std::vector<int16_t> result;
        // result.resize(m_width * m_height);

        for (int i = 0; i < m_width; ++i)
        {
            for (int j = 0; j < m_height; ++j)
            {
                getValue(result, i, j) = process(input, i, j);
            }
        }

        // for (int i = 0; i < m_width * m_height; ++i)
        // {
        //     result[i] = process(input[i]);
        // }

        return result;
    }

private:
    int process(const std::vector<uint8_t>& input, const int x, const int y)
    {
        
        // std::array<int, 25> arr;
        // int count = 0;
        // for (int i = x - 2; i <= x + 2; ++i)
        // {
        //     for (int j = y - 2; j <= y + 2; ++j)
        //     {
        //         arr[count++] = getValue(input, i, j);
        //     }
        // }

        // std::sort(arr.begin(), arr.end());

        // int avg = std::accumulate(arr.begin() + 10, arr.begin() + 15, 0) / 5;

        // return process(avg);
        

        return process(getValue(input, x, y));
    }

    int process(ushort Pos_z)
    {
        if (0 == Pos_z)
            return 0;
        
        Pos_z = 8 * Focus_Pixel * BaseLine / Pos_z;

        if (Pos_z > 26459)   Pos_z = 0.8358 * Pos_z + 4999.3;
		else if (Pos_z > 20449) Pos_z = 1.0109 * Pos_z + 288.24;
		else if (Pos_z > 16843) Pos_z = 0.8765 * Pos_z + 2639.4;
		else if (Pos_z > 8428) Pos_z = 1.0373 * Pos_z - 133.33;
		else if (Pos_z > 2400) Pos_z = 1.0174 * Pos_z - 15.134;
		else Pos_z = Pos_z;

        return Pos_z;
    }

    ushort getValue(const std::vector<uint8_t>& vec, int x, int y)
    {
        int index = 2 * (y * m_width + x);
        ushort *p = (ushort *)&vec[index];

        return *p;
        // return vec[y * m_width + x];
    }

    int32_t& getValue(std::vector<int32_t>& vec, int x, int y)
    {
        return vec[y * m_width + x];
    }

    const int m_width;
    const int m_height;

    const int   Focus_Pixel = 1180;      //定义双目统一后的像素焦距（大于左目和右目各自的焦距值），单位是像素。像素焦距 x 曝光芯片像元尺寸3.75 = 镜头焦距（单位为微米，除以1000得到单位为毫米的焦距值）
    const float BaseLine    = 180.29;  //定义相机双目基线，单位毫米，也就是双目瞳距


};
