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
python ./UniGO/data/synthetic_data/od_synthetic_data_generate.py --config ./UniGO/data/data_config/large_degroot.yaml
```

## Running Our Example

You can modify the config file to run the model example:

```
python main.py --config ./UniGO/config/example.yaml
```

The parameters are explained as follows:

- `result_dir`: results ## Results folder
- `ckpt_dir`: ckpt ## Checkpoint folder
- `logger_dir`: logger ## Logger folder
- `num_threads`: 24
- `seed`: 1234
- `num_workers`: 8
- `repeat`: 10 ## Number of times the experiment is repeated
- `run_multiple_splits`: []
- `data`:
  - `file_path_train`: null ## Automatically handled in the code
  - `file_path_test`: null ## Automatically handled in the code
  - `size`: 2000 ## Size of the synthetic dataset
  - `task_type`: 1 ## Task type: 1 for training on synthetic dataset, 2 for testing pre-trained model on a large synthetic dataset, 3 for testing pre-trained model on real-world dataset
  - `name`: degroot ## Name of the synthetic dataset
  - `split`:
    - 0.7
    - 0.1
    - 0.2
  - `batch_size`: 30
  - `train_timestep`: 1
- `train`:
  - `gpus`:
    - 0
  - `pretrained_ckpt`: null ## Load pre-trained model
  - `accelerator`: gpu
  - `max_epochs`: 1000
  - `min_epochs`: 50
  - `monitor`: val_wasserstein_distance
- `optim`:
  - `lr`: 0.01
  - `max_epoch`: 1000
  - `weight_decay`: 0.0001
  - `optimizer`: adamw
  - `lr_scheduler`: true
  - `scheduler`: step
  - `steps`: 100
- `model`:
  - `network_type`: mogo
  - `layer_type`: gcn
  - `loss_type`: mse
  - `gamma`: 0.5
  - `in_dim`: 1
  - `hidden_dim`: 8
  - `layers`: 2
  - `order`: 2
  - `dropout`: 0.0
  - `combine_mode`: mean ## Method for combining multi-order information
  - `force_layer`: 8
  - `num_class`: 4
  - `m`: 0.5
  - `num_node`: 2000
  - `cluster_type`: 0.01

## Acknowledgements

Open source codes involved in this project:

- pytorch-softdtw: https://github.com/Sleepwalking/pytorch-softdtw
- Pytorch-Lightning-Template: https://github.com/miracleyoo/pytorch-lightning-template
