import json
import numpy as np
import argparse
from os import path as osp
from tabulate import tabulate

def load_max_logs(dir_logs, nb_epochs=-1):
    path_logs = osp.join(dir_logs, 'logs.json')
    path_val_oe = osp.join(dir_logs, 'logs_eval_val_oe.json')
    logs = json.load(open(path_logs))
    out = {}
    epochs = logs["eval_epoch.epoch"]
    for k,v in logs.items():
        if 'eval_epoch' in k:
            argmax = np.argmax(v[:nb_epochs]) if nb_epochs != -1 else np.argmax(v)
            out[k] = epochs[argmax], v[argmax]
    if osp.isfile(path_val_oe):
        val_oe = json.load(open(path_val_oe))
        for k,v in val_oe.items():
            argmax = np.argmax(v[:nb_epochs]) if nb_epochs != -1 else np.argmax(v)
            out[k] = epochs[argmax], v[argmax]
    return out


def argmax(list_):
    return list_.index(max(list_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nb_epochs', default=-1, type=int)
    parser.add_argument('--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('--keys', type=str, nargs='*',
        default=['eval_epoch.epoch',
                 'eval_epoch.accuracy_top1',
                 'eval_epoch.overall',
                 'eval_epoch.map'])
    args = parser.parse_args()

    dir_logs = {}
    for raw in args.dir_logs:
        tmp = raw.split(':')
        if len(tmp) == 2:
            key, path = tmp
        elif len(tmp) == 1:
            path = tmp[0]
            key = osp.basename(path)
        else:
            raise ValueError(raw)
        dir_logs[key] = path

    keys = [key for key in args.keys]

    logs = {}
    for log_name in dir_logs.keys():
        logs[log_name] = load_max_logs(dir_logs[log_name], nb_epochs=args.nb_epochs)

    for key in keys:
        names = []
        values = []
        epochs = []
        for name, log_values in logs.items():
            if key in log_values:
                names.append(name)
                epoch, value = log_values[key]
                epochs.append(epoch)
                values.append(value)
        values_names = sorted(zip(values,names, epochs),reverse=True)
        values_names = [[i+1, name, value] for i, (value, name) in enumerate(values_names)]
        print('\n\n## {}\n'.format(key))
        print(tabulate(values_names, headers=['Place', 'Method', 'Score', 'Epoch']))


