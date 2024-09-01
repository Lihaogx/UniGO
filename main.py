import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml
import torch
import datetime
import argparse
import os
from model.model_interface import MInterface
from data.data_interface import DInterface
from omegaconf import OmegaConf
from utils import set_file_paths, add_test_phase_separator, load_config, save_all_results_to_csv, TimingCallback

def load_loggers(logger_dir, current_time_str, seed, run_id):
    loggers = [
        TensorBoardLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}'),
        CSVLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}')
    ]
    return loggers

def load_callbacks(ckpt_dir, current_time_str, seed, run_id, task_type, size, name):
    ckpt_path = f"{ckpt_dir}/{name}_{size}_{current_time_str}_{seed}_{run_id}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    callbacks = [
        plc.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=50,
            min_delta=0.001
        ),
        plc.ModelCheckpoint(
            monitor='val_wasserstein_distance',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_wasserstein_distance:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        plc.ModelCheckpoint(
            monitor='val_mse',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_mse:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        plc.ModelCheckpoint(
            monitor='val_accuracy',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_accuracy:.3f}',
            save_top_k=1,
            mode='max',
            save_last=True
        ),
        plc.ModelCheckpoint(
            monitor='val_f1',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_accuracy:.3f}',
            save_top_k=1,
            mode='max',
            save_last=True
        ),
        plc.ModelCheckpoint(
            monitor='val_dtw',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_accuracy:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        TimingCallback()
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
    all_results = []
    pre_results = []
    for run_id, seed in zip(*run_loop_settings(args)):
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        loggers = load_loggers(args.logger_dir, current_time_str, seed, run_id)
        callbacks = load_callbacks(args.ckpt_dir, current_time_str, seed, run_id, args.data.task_type, args.data.size, args.data.name)
        initial_ckpt_path = args.train.pretrained_ckpt
        
        data_module = DInterface(data_config=args.data)
        if args.data.task_type in [2,3]:
            model_module = MInterface.load_from_checkpoint(checkpoint_path=initial_ckpt_path, model_config=args.model, optim_config=args.optim)
        elif args.data.task_type == 1:
            model_module = MInterface(model_config=args.model, optim_config=args.optim)
        
        if args.data.task_type in [2,3]:
            args.train.max_epochs = 200
            args.train.min_epochs = 100
            trainer = Trainer(
                callbacks=callbacks, 
                logger=loggers, 
                max_epochs=args.train.max_epochs, 
                min_epochs=args.train.min_epochs, 
                accelerator=args.train.accelerator, 
                devices=args.train.gpus if args.train.gpus else None, 
                enable_checkpointing=True,
            )
            data_module.setup()
            pre_result = trainer.test(model=model_module, datamodule=data_module)
            for result in pre_result:
                result_entry = {
                    "run_id": run_id,
                    "seed": seed,
                    "current_time": current_time_str,
                    "best_checkpoint_path": None
                }
                result_entry.update(result)
                pre_results.append(result_entry)
            pre_result_file = f'{args.result_dir}/{args.data.task_type}/{args.model.layer_type}/pre_{args.data.name}_{args.model.layers}_{args.model.order}_{args.model.cluster_type}.csv'
            save_all_results_to_csv(pre_results, pre_result_file)
            
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
        trainer.fit(model_module, datamodule=data_module)

        # Determine the correct checkpoint for testing based on the monitoring target
        if args.train.monitor == 'val_wasserstein_distance':
            best_checkpoint_path = callbacks[1].best_model_path
        elif args.train.monitor == 'val_mse':
            best_checkpoint_path = callbacks[2].best_model_path
        elif args.train.monitor == 'val_accuracy':
            best_checkpoint_path = callbacks[3].best_model_path
        elif args.train.monitor == 'val_f1':
            best_checkpoint_path = callbacks[4].best_model_path
        elif args.train.monitor == 'val_dtw':
            best_checkpoint_path = callbacks[5].best_model_path
        else:
            raise ValueError("Unsupported monitoring target specified.")
        # Test using the best model after fitting
        test_results = trainer.test(model=model_module, datamodule=data_module)
        
        for result in test_results:
            result_entry = {
                "run_id": run_id,
                "seed": seed,
                "current_time": current_time_str,
                "best_checkpoint_path": best_checkpoint_path
            }
            result_entry.update(result)
            all_results.append(result_entry)
    if args.data.task_type == 1:
        if args.model.network_type == 'single':
            result_file = f'{args.result_dir}/{args.data.task_type}/{args.model.network_type}/{args.data.size}/{args.model.layer_type}/{args.data.name}_{args.model.force_layer}.csv'
        elif args.model.network_type == 'mogo':
            result_file = f'{args.result_dir}/{args.data.task_type}/{args.model.network_type}/{args.data.size}/{args.model.layer_type}/{args.data.name}_{args.model.layers}_{args.model.order}_{args.model.cluster_type}.csv'
    elif args.data.task_type in [2,3]:
        result_file = f'{args.result_dir}/{args.data.task_type}/{args.model.layer_type}/fine_{args.data.name}_{args.model.layers}_{args.model.order}_{args.model.cluster_type}.csv'
    save_all_results_to_csv(all_results, result_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    config = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    args = load_config(config.config)
    args = set_file_paths(args)
    main(args)
