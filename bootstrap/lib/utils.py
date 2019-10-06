import os
import random
import subprocess
import sys
import time

import numpy
import torch


def merge_dictionaries(dict1, dict2):
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dictionaries(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to


# to be able to reproduce exps on reload
def set_random_seed(seed):
    # is pytorch dataloader with multi-threads deterministic ?
    # cudnn may not be deterministic anyway
    torch.manual_seed(seed)  # on CPU and GPU
    numpy.random.seed(seed)  # useful ? not thread safe
    random.seed(seed)  # useful ? thread safe


def available_gpu_ids():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpu_ids = [int(idx) for idx in gpu_ids]
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
    return gpu_ids


def env_info():
    """ Collects information about the environment, for reproducibility """
    info = {}
    info['python_version'] = sys.version
    info['command'] = subprocess.list2cmdline(sys.argv)
    with open(os.devnull, 'w') as devnull:
        info['pip_modules'] = subprocess.check_output(['pip', 'freeze'], stderr=devnull)
        try:
            info['git_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                                         stderr=devnull).strip().decode('UTF-8')
            info['git_local_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=devnull)
            info['git_status'] = subprocess.check_output(['git', 'status'], stderr=devnull)
            info['git_origin_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'origin/{}'.format(info['git_branch'])], stderr=devnull)
            info['git_diff_origin_commit'] = subprocess.check_output(
                ['git', 'diff', 'origin/{}'.format(info['git_branch'])], stderr=devnull)
            info['git_log_since_origin'] = subprocess.check_output(
                ['git', 'log', '--pretty=oneline', '{}..HEAD'.format(info['git_origin_commit'])], stderr=devnull)
        except subprocess.CalledProcessError:
            pass
    info['creation_time'] = time.strftime('%y-%m-%d-%H-%M-%S')
    info['sysname'] = os.uname()[0]
    info['nodename'] = os.uname()[1]
    info['release'] = os.uname()[2]
    info['version'] = os.uname()[3]
    info['architecture'] = os.uname()[4]
    info['user'] = os.environ['USER']
    info['path'] = os.environ['PWD']
    info['conda_environment'] = os.environ.get('CONDA_DEFAULT_ENV', '')
    info['environment_vars'] = dict(os.environ)
    info = {k: info[k].decode("UTF-8") if type(info[k]) == bytes else info[k] for k in info}
    return info
