import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml
import torch
import datetime
import argparse
import os
from model.model_interface_sinn import MInterface
from data.data_interface_sinn import DInterface
from utils import set_file_paths, load_config



def load_loggers(logger_dir, current_time_str, seed, run_id):
    loggers = [
        TensorBoardLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}'),
        CSVLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}')
    ]
    return loggers

def load_callbacks(ckpt_dir, current_time_str, seed, run_id, task_type):
    # 构建存储模型检查点的目录路径
    ckpt_path = f"{ckpt_dir}/{current_time_str}_{seed}_{run_id}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    callbacks = [
        plc.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=50,
            min_delta=0.001
        ),
        plc.ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_wasserstein_distance:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
    ]
    if task_type == 3:  # Assuming task 3 requires learning rate adjustments
        callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    return callbacks

def run_loop_settings(args):
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random seed is reset to the initial cfg.seed value for each run iteration.
    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(args.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [args.seed + x for x in range(num_iterations)]
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(args.run_multiple_splits)
        seeds = [args.seed] * num_iterations
    return run_ids, seeds


def main(args):
    pl.seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.set_default_dtype(torch.float32)
    
    for run_id, seed in zip(*run_loop_settings(args)):
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        loggers = load_loggers(args.logger_dir, current_time_str, seed, run_id)
        callbacks = load_callbacks(args.ckpt_dir, current_time_str, seed, run_id, args.data.task_type)

        data_module = DInterface(data_config=args.data)
        if args.model.num_users == 0:
            data_module.setup()
            for batch in data_module.train_dataloader():
                args.model.num_users = batch['num_users'][0]
        model_module = MInterface(model_config=args.model, optim_config=args.optim)
        
        # Handle task type 3 special case for loading pretrained model
        initial_ckpt_path = None
        
            
        trainer = Trainer(
            callbacks=callbacks, 
            logger=loggers, 
            max_epochs=args.train.max_epochs, 
            min_epochs=args.train.min_epochs, 
            accelerator=args.train.accelerator, 
            devices=args.train.gpus if args.train.gpus else None, 
            enable_checkpointing=True,
        )

        # Fit model
        trainer.fit(model_module, datamodule=data_module, ckpt_path=initial_ckpt_path)

        # Determine the correct checkpoint for testing based on the monitoring target
        best_checkpoint_path = callbacks[1].best_model_path

        trainer.test(model=model_module, datamodule=data_module, ckpt_path=best_checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default="/home/lh/MOGO/config/sinn_abortion.yaml", required=True,
                        help='Path to the configuration file')
    config = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    args = load_config(config.config)
    args = set_file_paths(args)
    main(args)
