/*******************************************************
 文件名：ImagePreProcessGPU_kernels.cu
 作者：sharkls
 描述：GPU加速的图像预处理CUDA核函数
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// CUDA内核函数定义
__global__ void resizeKernel(uchar3* src, uchar3* dst, int src_width, int src_height, 
                            int dst_width, int dst_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_width && y < dst_height) {
        float src_x = (float)x * src_width / dst_width;
        float src_y = (float)y * src_height / dst_height;
        
        int src_x0 = (int)src_x;
        int src_y0 = (int)src_y;
        int src_x1 = min(src_x0 + 1, src_width - 1);
        int src_y1 = min(src_y0 + 1, src_height - 1);
        
        float fx = src_x - src_x0;
        float fy = src_y - src_y0;
        
        uchar3 p00 = src[src_y0 * src_width + src_x0];
        uchar3 p01 = src[src_y0 * src_width + src_x1];
        uchar3 p10 = src[src_y1 * src_width + src_x0];
        uchar3 p11 = src[src_y1 * src_width + src_x1];
        
        uchar3 result;
        result.x = (unsigned char)((1 - fx) * (1 - fy) * p00.x + fx * (1 - fy) * p01.x + 
                          (1 - fx) * fy * p10.x + fx * fy * p11.x);
        result.y = (unsigned char)((1 - fx) * (1 - fy) * p00.y + fx * (1 - fy) * p01.y + 
                          (1 - fx) * fy * p10.y + fx * fy * p11.y);
        result.z = (unsigned char)((1 - fx) * (1 - fy) * p00.z + fx * (1 - fy) * p01.z + 
                          (1 - fx) * fy * p10.z + fx * fy * p11.z);
        
        dst[y * dst_width + x] = result;
    }
}

__global__ void normalizeKernel(uchar3* src, float* dst, int width, int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = src[idx];
        
        // 注意：输入已经是RGB格式（经过BGR到RGB转换）
        dst[idx * 3 + 0] = pixel.x * scale;  // R
        dst[idx * 3 + 1] = pixel.y * scale;  // G
        dst[idx * 3 + 2] = pixel.z * scale;  // B
    }
}

__global__ void hwcToChwKernel(float* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int hwc_idx = (y * width + x) * 3;
        int chw_idx = y * width + x;
        
        dst[0 * height * width + chw_idx] = src[hwc_idx + 0];  // R -> C0
        dst[1 * height * width + chw_idx] = src[hwc_idx + 1];  // G -> C1
        dst[2 * height * width + chw_idx] = src[hwc_idx + 2];  // B -> C2
    }
}

__global__ void padImageKernel(float* src, float* dst, int src_width, int src_height, 
                              int dst_width, int dst_height, int pad_top, int pad_left, 
                              float pad_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_width && y < dst_height) {
        int dst_idx = y * dst_width + x;
        
        // 检查是否在填充区域
        if (x < pad_left || x >= pad_left + src_width || 
            y < pad_top || y >= pad_top + src_height) {
            // 填充区域，设置为填充值
            dst[dst_idx] = pad_value;
        } else {
            // 图像区域，复制源数据
            int src_x = x - pad_left;
            int src_y = y - pad_top;
            int src_idx = src_y * src_width + src_x;
            dst[dst_idx] = src[src_idx];
        }
    }
}

__global__ void bgrToRgbKernel(uchar3* bgr, uchar3* rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = bgr[idx];
        
        // BGR to RGB conversion
        rgb[idx].x = pixel.z;  // B -> R
        rgb[idx].y = pixel.y;  // G -> G
        rgb[idx].z = pixel.x;  // R -> B
    }
}

// 核函数启动器（C接口，供C++调用）
extern "C" {
    void launchResizeKernel(uchar3* src, uchar3* dst, int src_width, int src_height, 
                           int dst_width, int dst_height, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x, 
                      (dst_height + block_size.y - 1) / block_size.y);
        
        resizeKernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, src_width, src_height, dst_width, dst_height);
    }
    
    void launchNormalizeKernel(uchar3* src, float* dst, int width, int height, 
                              float scale, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                      (height + block_size.y - 1) / block_size.y);
        
        normalizeKernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, width, height, scale);
    }
    
    void launchHWCtoCHWKernel(float* src, float* dst, int width, int height, 
                             cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                      (height + block_size.y - 1) / block_size.y);
        
        hwcToChwKernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, width, height);
    }
    
    void launchPadImageKernel(float* src, float* dst, int src_width, int src_height, 
                             int dst_width, int dst_height, int pad_top, int pad_left, 
                             float pad_value, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x, 
                      (dst_height + block_size.y - 1) / block_size.y);
        
        padImageKernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, src_width, src_height, dst_width, dst_height, 
            pad_top, pad_left, pad_value);
    }
    
    void launchBgrToRgbKernel(uchar3* bgr, uchar3* rgb, int width, int height, 
                             cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                      (height + block_size.y - 1) / block_size.y);
        
        bgrToRgbKernel<<<grid_size, block_size, 0, stream>>>(
            bgr, rgb, width, height);
    }
} 