import json
import numpy as np
import argparse
from os import path as osp
from tabulate import tabulate


def load_values(dir_logs, metrics, nb_epochs=-1, best=None):
    json_files = {}
    values = {}

    # load argsup of best
    if best:
        if best['json'] not in json_files:
            with open(osp.join(dir_logs, f'{best["json"]}.json')) as f:
                json_files[best['json']] = json.load(f)

        jfile = json_files[best['json']]
        vals = jfile[best['name']]
        end = len(vals) if nb_epochs == -1 else nb_epochs
        argsup = np.__dict__[f'arg{best["order"]}'](vals[:end])

    # load logs
    for mkey, metric in metrics.items():
        # open json_files
        if metric['json'] not in json_files:
            with open(osp.join(dir_logs, f'{metric["json"]}.json')) as f:
                json_files[metric['json']] = json.load(f)

        jfile = json_files[metric['json']]

        if 'train' in metric['name']:
            epoch_key = 'train_epoch.epoch'
        else:
            epoch_key = 'eval_epoch.epoch'

        if epoch_key in jfile:
            epochs = jfile[epoch_key]
        else:
            epochs = jfile['epoch']

        vals = jfile[metric['name']]
        if not best:
            end = len(vals) if nb_epochs == -1 else nb_epochs
            argsup = np.__dict__[f'arg{metric["order"]}'](vals[:end])

        try:
            values[metric['name']] = epochs[argsup], vals[argsup]
        except:
            values[metric['name']] = epochs[argsup-1], vals[argsup-1]
    return values

def main(args):
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

    metrics = {}
    for json, name, order in args.metrics:
        metrics[f'{json}_{name}'] = {
            'json': json,
            'name': name,
            'order': order
        }

    if args.best:
        json, name, order = args.best
        best = {
            'json': json,
            'name': name,
            'order': order
        }
    else:
        best = None

    logs = {}
    for name, dir_log in dir_logs.items():
        logs[name] = load_values(dir_log, metrics,
            nb_epochs=args.nb_epochs,
            best=best)

    for mkey, metric in metrics.items():
        names = []
        values = []
        epochs = []
        for name, vals in logs.items():
            if metric['name'] in vals:
                names.append(name)
                epoch, value = vals[metric['name']]
                epochs.append(epoch)
                values.append(value)
        if values:
            values_names = sorted(zip(values, names, epochs), reverse=metric['order']=='max')
            values_names = [[i + 1, name, value, epoch] for i, (value, name, epoch) in enumerate(values_names)]
            print('\n\n## {}\n'.format(metric['name']))
            print(tabulate(values_names, headers=['Place', 'Method', 'Score', 'Epoch']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-m', '--metrics', type=str, action='append', nargs=3,
                        metavar=('json', 'name', 'order'),
                        default=[['logs', 'eval_epoch.accuracy_top1', 'max'],
                                 ['logs', 'eval_epoch.accuracy_top5', 'max'],
                                 ['logs', 'eval_epoch.loss', 'min']])
    parser.add_argument('-b', '--best', type=str, nargs=3,
                        metavar=('json', 'name', 'order'),
                        default=['logs', 'eval_epoch.accuracy_top1', 'max'])
    args = parser.parse_args()
    main(args)
