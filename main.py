#!/usr/bin/env python3
import os
import click
import traceback

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import bootstrap.models as models
import bootstrap.lib.utils as utils
import bootstrap.datasets as datasets
import bootstrap.engines as engines
import bootstrap.optimizers as optimizers

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options


def init_experiment_directory(exp_dir, resume=None):
    # create the experiment directory
    if not os.path.isdir(exp_dir):
        os.system('mkdir -p '+exp_dir)
    else:
        if resume is None:
            if click.confirm('Exp directory already exists in {}. Erase?'
                    .format(exp_dir, default=False)):
                os.system('rm -r '+exp_dir)
                os.system('mkdir -p '+exp_dir)
            else:
                os._exit(1)

    # if resume to evaluate the model on one epoch
    if resume and Options()['dataset']['train_split'] is None:
        eval_split = Options()['dataset']['eval_split']
        path_yaml = os.path.join(exp_dir, 'options_eval_{}.yaml'.format(eval_split))
        logs_name = 'logs_eval_{}'.format(eval_split)
    else:
        path_yaml = os.path.join(exp_dir, 'options.yaml')
        logs_name = 'logs'
        
    # create the options.yaml file
    Options.save_yaml_opts(path_yaml)

    # open(write) the logs.txt file and init logs.json path for later
    Logger(exp_dir, name=logs_name)


def main(path_opts=None):
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    Options(path_opts)

    # make exp directory if resume=None
    init_experiment_directory(Options()['exp']['dir'], Options()['exp']['resume'])

    # initialiaze seeds to be able to reproduce experiment on reload
    utils.set_random_seed(Options()['misc']['seed'])

    # display and save options as yaml file in exp dir
    Logger().log_dict('options', Options(), should_print=True)

    # display server name and GPU(s) id(s)
    Logger()(os.uname())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        Logger()('CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

    # engine can train, eval, optimize the model
    # engine can save and load the model and optimizer
    engine = engines.factory()

    # dataset is a dictionnary which contains all the needed datasets indexed by splits
    # (example: dataset.keys() -> ['train','val'])
    dataset = datasets.factory()
    engine.dataset = dataset

    # model includes a network, a criterion and a metric
    # model can register engine hooks (begin epoch, end batch, end batch, etc.)
    # (example: "calculate mAP at the end of the evaluation epoch")
    model = models.factory(engine)
    engine.model = model

    # optimizer can register engine hooks
    optimizer = optimizers.factory(model, engine)
    engine.optimizer = optimizer    

    # load the model and optimizer from a checkpoint
    if Options()['exp']['resume']:
        engine.resume()

    # if no training split, evaluate the model on the evaluation split
    # (example: $ python main.py --dataset.train_split --dataset.eval_split test)
    engine.eval()
  
    # optimize the model on the training split for several epochs
    # (example: $ python main.py --dataset.train_split train)
    # if evaluation split, evaluate the model after each epochs
    # (example: $ python main.py --dataset.train_split train --dataset.eval_split val)
    engine.train()
        

if __name__ == '__main__':
    try:
        main()
    # to avoid traceback for -h flag in arguments line
    except SystemExit:
        pass
    except:
        # to be able to write the error trace to exp_dir/logs.txt
        try:
            Logger()(traceback.format_exc(), Logger.ERROR)
        except:
            pass
