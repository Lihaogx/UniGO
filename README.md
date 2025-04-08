# UniGO

Implementation for the paper "UniGO: A Multi-Order Graph Neural Network for Learning Opinion Dynamics on Graphs."

## Requirements

This project is recommended to run in a Python 3.9 environment. The necessary packages are listed in `requirements.txt` and can be installed using the following command:

```
pip install -r requirements.txt
```

Please note that `torch_cluster`, `torch_scatter`, and `torch_sparse` required by `pytorch-geometric` are not included in the requirements because their versions are CUDA-dependent. It is recommended to choose the appropriate versions from [here](https://pytorch-geometric.com/whl/).

## Synthetic Data Generation

This project provides a method for generating synthetic data. The command to run it is:

```
python ./UniGO/data/synthetic_data/od_synthetic_data_generate.py
```
In od_synthetic_data_generate.py, we have built-in a parameter grid to control various parameters in the dataset. The dataset parameters are defined in the base_config within the file.

## Real Data Generation

The command to run it is:

```
python ./UniGO/data/real_data/od_real_data_generate.py
```
This code will automatically process the data in ./UniGO/data/raw_data. Note that the raw data needs to be formatted as separate txt files for nodes and opinions.

## Running Our Example

You can modify the config file to run the model example:

```
python main.py --config ./UniGO/config/unigo_sage.yaml
```
## Citation

If you find this work useful in your research, please consider citing:

```
@article{li2025unigo,
  title={UniGO: A Unified Graph Neural Network for Modeling Opinion Dynamics on Graphs},
  author={Li, Hao and Jiang, Hao and Zheng, Yuke and Sun, Hao and Gong, Wenying},
  journal={arXiv preprint arXiv:2502.11519},
  year={2025}
}
```
