import json
import numpy as np
import argparse
from os import path as osp
from tabulate import tabulate

def load_max_logs(dir_logs, metrics, nb_epochs=-1):
    path_logs = osp.join(dir_logs, 'logs.json')
    path_val_oe = osp.join(dir_logs, 'logs_eval_val_oe.json')
    logs = json.load(open(path_logs))
    out = {}
    epochs = logs["eval_epoch.epoch"]
    for k,v in logs.items():
        if k in metrics:
            if metrics[k] == "max":
                argsup = np.argmax(v[:nb_epochs]) if nb_epochs != -1 else np.argmax(v)
            elif metrics[k] == "min":
                argsup = np.argmin(v[:nb_epochs]) if nb_epochs != -1 else np.argmin(v)
            out[k] = epochs[argsup], v[argsup]
    if osp.isfile(path_val_oe):
        val_oe = json.load(open(path_val_oe))
        for k,v in val_oe.items():
            if metrics[k] == "max":
                argsup = np.argmax(v[:nb_epochs]) if nb_epochs != -1 else np.argmax(v)
            elif metrics[k] == "min":
                argsup = np.argmin(v[:nb_epochs]) if nb_epochs != -1 else np.argmin(v)
            out[k] = epochs[argsup], v[argsup]
    return out


def argmax(list_):
    return list_.index(max(list_))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-k', '--keys', type=str, action='append', nargs=2,
                        metavar=('metric', 'order'),
                        default=[['eval_epoch.epoch', 'max'],
                                 ['eval_epoch.accuracy_top1', 'max'],
                                 ['eval_epoch.overall', 'max'],
                                 ['eval_epoch.map', 'max']])
    args = parser.parse_args()

    dir_logs = {}
    for raw in args.dir_logs:
        tmp = raw.split(':')
        if len(tmp) == 2:
            key, path = tmp
        elif len(tmp) == 1:
            path = tmp[0]
            key = osp.basename(osp.normpath(path))
        else:
            raise ValueError(raw)
        dir_logs[key] = path

    metrics = {key: min_or_max for key, min_or_max in args.keys}

    logs = {}
    for log_name in dir_logs.keys():
        logs[log_name] = load_max_logs(dir_logs[log_name], metrics, nb_epochs=args.nb_epochs)

    for key in metrics:
        names = []
        values = []
        epochs = []
        for name, log_values in logs.items():
            if key in log_values:
                names.append(name)
                epoch, value = log_values[key]
                epochs.append(epoch)
                values.append(value)
        if values:
            values_names = sorted(zip(values, names, epochs),reverse=True)
            values_names = [[i+1, name, value, epoch] for i, (value, name, epoch) in enumerate(values_names)]
            print('\n\n## {}\n'.format(key))
            print(tabulate(values_names, headers=['Place', 'Method', 'Score', 'Epoch']))


if __name__ == '__main__':
    main()
