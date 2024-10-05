import h5py
import os
from tqdm import tqdm

def filter_h5_file(input_file, output_file):
    """
    从输入的 HDF5 文件中筛选出满足条件的数据项，并保存到新的 HDF5 文件中。

    参数：
    - input_file：原始 HDF5 文件的路径
    - output_file：筛选后的 HDF5 文件的路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 打开输入的 HDF5 文件（只读模式）
    with h5py.File(input_file, 'r') as h5_in:
        # 获取所有顶层组的键
        all_keys = list(h5_in.keys())
        num_items = len(all_keys)

        print(f"总共发现 {num_items} 个数据项。开始筛选...")

        # 创建输出的 HDF5 文件（写模式，如果已存在则覆盖）
        with h5py.File(output_file, 'w') as h5_out:
            num_selected = 0  # 计数器，记录保留的数据项数量
            # 使用 tqdm 显示进度条
            for key in tqdm(all_keys, desc="筛选数据项"):
                data_group = h5_in[key]
                # 获取 convergence_step 属性
                convergence_step = data_group.attrs.get('convergence_step', None)
                if convergence_step is None:
                    continue  # 如果没有 convergence_step 属性，跳过该数据项
                # 检查条件：convergence_step == -1 或 convergence_step > 50
                if convergence_step == -1 or convergence_step > 50:
                    # 将满足条件的组复制到输出文件中
                    h5_in.copy(key, h5_out)
                    num_selected += 1  # 计数器加1
        print(f"筛选完成，共保留了 {num_selected} 个数据项，结果已保存到 {output_file}")

if __name__ == '__main__':
    input_file = '/home/lh/UniGO/data/synthetic.h5'
    output_file = '/home/lh/UniGO/data/synthetic_filtered.h5'
    filter_h5_file(input_file, output_file)
