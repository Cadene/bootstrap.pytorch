import io
import os
import sys
import click
import torch
import traceback
import torch.backends.cudnn as cudnn

# This line is necessary for profiling
__package__ = 'bootstrap'

from .lib import utils
from .lib.logger import Logger
from .lib.options import Options
from . import engines
from . import datasets
from . import models
from . import optimizers
from . import views


def init_experiment_directory(exp_dir, resume=None):
    # create the experiment directory
    if not os.path.isdir(exp_dir):
        os.system('mkdir -p ' + exp_dir)
    else:
        if resume is None:
            if click.confirm('Exp directory already exists in {}. Erase?'
                    .format(exp_dir, default=False)):
                os.system('rm -r ' + exp_dir)
                os.system('mkdir -p ' + exp_dir)
            else:
                os._exit(1)


def init_logs_options_files(exp_dir, resume=None):
    # get the logs name which is used for the txt, json and yaml files
    # default is `logs.txt`, `logs.json` and `options.yaml`
    if 'logs_name' in Options()['misc'] and Options()['misc']['logs_name'] is not None:
        logs_name = 'logs_{}'.format(Options()['misc']['logs_name'])
        path_yaml = os.path.join(exp_dir, 'options_{}.yaml'.format(logs_name))
    elif resume and Options()['dataset']['train_split'] is None:
        eval_split = Options()['dataset']['eval_split']
        path_yaml = os.path.join(exp_dir, 'options_eval_{}.yaml'.format(eval_split))
        logs_name = 'logs_eval_{}'.format(eval_split)
    else:
        path_yaml = os.path.join(exp_dir, 'options.yaml')
        logs_name = 'logs'

    # create the options.yaml file
    if not os.path.isfile(path_yaml):
        Options().save(path_yaml)

    # create the logs.txt and logs.json files
    Logger(exp_dir, name=logs_name)


def run(path_opts=None):
    # initialiaze seeds to be able to reproduce experiment on reload
    utils.set_random_seed(Options()['misc']['seed'])

    Logger().log_dict('options', Options(), should_print=True) # display options
    Logger()(os.uname()) # display server name

    if torch.cuda.is_available():
        cudnn.benchmark = True
        Logger()('Available GPUs: {}'.format(utils.available_gpu_ids()))

    # engine can train, eval, optimize the model
    # engine can save and load the model and optimizer
    engine = engines.factory()

    # dataset is a dictionary that contains all the needed datasets indexed by modes
    # (example: dataset.keys() -> ['train','eval'])
    engine.dataset = datasets.factory(engine)

    # model includes a network, a criterion and a metric
    # model can register engine hooks (begin epoch, end batch, end batch, etc.)
    # (example: "calculate mAP at the end of the evaluation epoch")
    # note: model can access to datasets using engine.dataset
    engine.model = models.factory(engine)

    # optimizer can register engine hooks
    engine.optimizer = optimizers.factory(engine.model, engine)

    # view will save a view.html in the experiment directory
    # with some nice plots and curves to monitor training
    engine.view = views.factory(engine)

    # load the model and optimizer from a checkpoint
    if Options()['exp']['resume']:
        engine.resume()

    # if no training split, evaluate the model on the evaluation split
    # (example: $ python main.py --dataset.train_split --dataset.eval_split test)
    if not Options()['dataset']['train_split']:
        engine.eval()

    # optimize the model on the training split for several epochs
    # (example: $ python main.py --dataset.train_split train)
    # if evaluation split, evaluate the model after each epochs
    # (example: $ python main.py --dataset.train_split train --dataset.eval_split val)
    if Options()['dataset']['train_split']:
        engine.train()


def activate_debugger(): 
    if Options()['misc'].get('debug', False):
        Logger().set_level(Logger.DEBUG)
        Logger()('Debug mode activated.', log_level=Logger.DEBUG)


def activate_profiler():
    if sys.version_info[0] == 3: #PY3
        import builtins
    else:
        import __builtin__ as builtins
    if Options()['misc'].get('profile', False):
        # if profiler is activated, associate line_profiler
        Logger()('Activating line_profiler...')
        try:
            import line_profiler
        except ModuleNotFoundError:
            Logger()('Failed to import line_profiler.', log_level=Logger.ERROR, raise_error=False)
            Logger()('Please install it from https://github.com/rkern/line_profiler', log_level=Logger.ERROR, raise_error=False)
            return
        prof = line_profiler.LineProfiler()
        builtins.__dict__['profile'] = prof
    else:
        # otherwise, create a blank profiler, to disable profiling code
        builtins.__dict__['profile'] = lambda func: func
        prof = None
    return prof


def process_profiler(prof):
     if Options()['misc'].get('profile', False):
        # unavoidable lazy import -- only if profiler is enabled
        from line_profiler import show_text
        results_file = os.path.join(Options()['exp']['dir'], 'profile_results.lprof')
        Logger()('Saving profiler results to {}...'.format(results_file))
        prof.dump_stats(results_file)
        stats = prof.get_stats()
        textio = io.StringIO()
        show_text(stats.timings, stats.unit, stream=textio)
        lines = textio.getvalue()
        Logger()('Printing profiling results', log_level=Logger.SYSTEM)
        for line in lines.splitlines():
            Logger()(line, log_level=Logger.SYSTEM, print_header=False)


def process_debugger():
    if Options()['misc'].get('debug', False):
        import pdb
        pdb.post_mortem()


def main(path_opts=None, run=None):
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    Options(path_opts)
    # init options and exp dir for logging
    init_experiment_directory(Options()['exp']['dir'], Options()['exp']['resume'])
    init_logs_options_files(Options()['exp']['dir'], Options()['exp']['resume'])
    # activate debugger if enabled
    activate_debugger()
    # activate profiler if enabled
    profiler = activate_profiler()
    # run bootstrap routine
    try:
        run(path_opts=path_opts)
    except SystemExit:
        # to avoid traceback for -h flag in arguments line
        pass
    except KeyboardInterrupt:
        Logger()('KeyboardInterrupt signal received. Exiting...', log_level=Logger.ERROR, raise_error=False)
    except:
        # to be able to write the error trace to exp_dir/logs.txt
        try:
            Logger()(traceback.format_exc(), log_level=Logger.ERROR)
        except:
            print('Failed to call Logger for the following stack trace:')
            print(traceback.format_exc())
            pass
        process_debugger()
    finally:
        # write profiling results, if enabled
        process_profiler(profiler)


if __name__ == '__main__':
    main(run=run)

