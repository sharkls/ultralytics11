import numpy as np
import argparse

def txt_to_npy(txt_path, npy_path):
    # 读取txt文件，假设每行是空格或逗号分隔的数字
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        matrix = []
        for line in lines:
            # 支持空格或逗号分隔
            nums = [float(x) for x in line.replace(',', ' ').split()]
            matrix.append(nums)
        matrix = np.array(matrix)
    # 保存为npy
    np.save(npy_path, matrix)
    print(f"已将 {txt_path} 转换为 {npy_path}")

def main():
    parser = argparse.ArgumentParser(description="将单应性矩阵txt文件转换为npy格式")
    parser.add_argument('--txt_path', type=str, default='/ultralytics/data/Myself-v3/extrinsics/test/001624.txt', help='输入的txt文件路径')
    parser.add_argument('--npy_path', type=str, default='./runs/mapping_matrix-v3/manual_homography_matrix-v3.npy', help='输出的npy文件路径')
    args = parser.parse_args()
    txt_to_npy(args.txt_path, args.npy_path)

if __name__ == '__main__':
    main()