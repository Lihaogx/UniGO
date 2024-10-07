import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import gc
import h5py

def process_large_pkl_files(file_list, hdf5_file, batch_size=100):
    """
    处理多个大型的 .pkl 文件，分批次读取和处理，将结果写入单个 HDF5 文件中
    """
    os.makedirs(os.path.dirname(hdf5_file), exist_ok=True)
    idx = 0  # 用于跟踪 HDF5 中的数据索引

    try:
        # 打开 HDF5 文件进行写入
        with h5py.File(hdf5_file, 'w') as h5f:
            for file_path in file_list:
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'rb') as f:
                        # 假设每个文件是一个包含多个 data_item 的列表
                        data_list = pickle.load(f)

                        num_items = len(data_list)
                        num_batches = (num_items + batch_size - 1) // batch_size

                        # 使用 tqdm 显示文件的处理进度
                        for batch_num in tqdm(range(num_batches), desc=f"Processing {os.path.basename(file_path)}"):
                            batch_data_list = data_list[batch_num * batch_size : (batch_num + 1) * batch_size]

                            for data_item in batch_data_list:
                                try:
                                    if data_item.convergence_step == -1 or data_item.convergence_step > 80:
                                        # 重建时间序列
                                        x = data_item.x  # [36, 1000, 15]
                                        y = data_item.y  # [36, 1000, 50]
                                        x0 = x[0]  # [1000, 15]
                                        y0 = y[0]  # [1000, 50]
                                        y35 = y[35]  # [1000, 50]
                                        y35_last35 = y35[:, -35:]  # [1000, 35]
                                        time_series = torch.cat([x0, y0, y35_last35], dim=1)  # [1000, 100]

                                        if time_series.shape[1] != 100:
                                            continue  # 跳过该数据项

                                        x_input = time_series[:, :10]  # [1000, 10]
                                        y_output = time_series[:, 10:]  # [1000, 90]

                                        # 更新 data_item 的 x 和 y
                                        data_item.x = x_input
                                        data_item.y = y_output

                                        # 写入 HDF5 文件
                                        group_name = f'data_{idx}'
                                        group = h5f.create_group(group_name)
                                        group.create_dataset('x', data=data_item.x.numpy(), compression="gzip")
                                        group.create_dataset('y', data=data_item.y.numpy(), compression="gzip")
                                        group.create_dataset('edge_index', data=data_item.edge_index.numpy(), compression="gzip")
                                        group.create_dataset('cluster_node_indices', data=data_item.cluster_node_indices.numpy(), compression="gzip")
                                        group.create_dataset('cluster_ptr', data=data_item.cluster_ptr.numpy(), compression="gzip")
                                        group.attrs['convergence_step'] = data_item.convergence_step
                                        idx += 1

                                        # 释放内存
                                        del data_item, x, y
                                        gc.collect()
                                except Exception as e:
                                    print(f"Error processing data item {idx}: {e}")
                                    continue  # 继续处理下一个数据项

                            # 批次处理完毕，释放内存
                            del batch_data_list
                            gc.collect()

                    # 处理完一个文件，释放内存
                    del data_list
                    gc.collect()

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        print("All files have been processed.")
    except Exception as e:
        print(f"Error opening/writing HDF5 file: {e}")

if __name__ == "__main__":
    file_list = [
        '/home/lh/UniGO/data/processed_data_batch_1.pkl',
        '/home/lh/UniGO/data/processed_data_batch_2.pkl',
        '/home/lh/UniGO/data/processed_data_batch_3.pkl',
        '/home/lh/UniGO/data/processed_data_batch_4.pkl',
        '/home/lh/UniGO/data/processed_data_batch_5.pkl',
        '/home/lh/UniGO/data/processed_data_batch_6.pkl',
        '/home/lh/UniGO/data/processed_data_batch_7.pkl',
        '/home/lh/UniGO/data/processed_data_batch_8.pkl'
    ]
    hdf5_file = '/home/lh/UniGO/data/synthetic_v2.h5'
    process_large_pkl_files(file_list, hdf5_file, batch_size=100)
