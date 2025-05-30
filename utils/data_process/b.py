import os
import argparse
from pathlib import Path

def process_label_file(file_path):
    """处理单个标签文件，将0和1互换"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            # 第一个数字是类别，如果是0或1则互换
            if parts[0] == '0':
                parts[0] = '1'
            elif parts[0] == '1':
                parts[0] = '0'
        new_lines.append(' '.join(parts) + '\n')
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def main():
    parser = argparse.ArgumentParser(description='处理标签文件，将指定范围内的0和1标签互换')
    parser.add_argument('--path', type=str, default='data/Data/0529/0528all/labels/visible1/train',
                      help='标签文件所在的基础路径')
    parser.add_argument('--start_file', type=str, default='13668.txt',
                      help='起始文件名')
    parser.add_argument('--end_file', type=str, default='15485.txt',
                      help='结束文件名')
    
    args = parser.parse_args()
    
    # 获取所有txt文件
    base_path = Path(args.path)
    all_txt_files = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.txt'):
                all_txt_files.append(os.path.join(root, file))
    
    # 按文件名排序
    all_txt_files.sort()
    
    # 处理指定范围内的文件
    for file_path in all_txt_files:
        file_name = os.path.basename(file_path)
        if args.start_file <= file_name <= args.end_file:
            print(f'处理文件: {file_path}')
            process_label_file(file_path)

if __name__ == '__main__':
    main()
