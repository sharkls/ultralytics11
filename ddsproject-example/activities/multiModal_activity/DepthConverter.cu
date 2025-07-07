#include "DepthConverter.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

// CUDA核函数
__device__ int process_depth(ushort Pos_z, int Focus_Pixel, float BaseLine) {
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

__global__ void depth_kernel(
    const uint8_t* input, int32_t* output, int width, int height, int Focus_Pixel, float BaseLine)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int arr[49];
    int count = 0;
    for (int dx = -3; dx <= 3; ++dx) {
        int xx = min(max(x + dx, 0), width - 1);
        for (int dy = -3; dy <= 3; ++dy) {
            int yy = min(max(y + dy, 0), height - 1);
            int idx = 2 * (yy * width + xx);
            ushort val = *(ushort*)(input + idx);
            arr[count++] = val;
        }
    }
    // 简单冒泡排序
    for (int i = 0; i < 48; ++i)
        for (int j = i + 1; j < 49; ++j)
            if (arr[i] > arr[j]) { int t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    int avg = 0;
    for (int i = 21; i < 28; ++i) avg += arr[i];
    avg /= 7;

    int depth = process_depth(avg, Focus_Pixel, BaseLine);
    output[y * width + x] = depth;
}

void DepthConverter::process_gpu(const std::vector<uint8_t>& input, std::vector<int32_t>& result)
{
    uint8_t* d_input = nullptr;
    int32_t* d_output = nullptr;
    size_t input_size = m_width * m_height * 2;
    size_t output_size = m_width * m_height * sizeof(int32_t);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((m_width + 15) / 16, (m_height + 15) / 16);

    depth_kernel<<<grid, block>>>(d_input, d_output, m_width, m_height, Focus_Pixel, BaseLine);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
} 