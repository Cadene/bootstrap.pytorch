import json
import argparse
from os import path as osp
from tabulate import tabulate

def load_max_logs(dir_logs, nb_epochs=-1):
    path_logs = osp.join(dir_logs, 'logs.json')
    path_val_oe = osp.join(dir_logs, 'logs_eval_val_oe.json')
    logs = json.load(open(path_logs))
    out = {}
    for k,v in logs.items():
        if 'eval_epoch' in k:
            if nb_epochs != -1:
                out[k] = max(v[:nb_epochs])
            else:
                out[k] = max(v)
    if osp.isfile(path_val_oe):
        val_oe = json.load(open(path_val_oe))
        for k,v in val_oe.items():
            if nb_epochs != -1:
                out[k] = max(v[:nb_epochs])
            else:
                out[k] = max(v)
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
                 'val_epoch.overall'])
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
    #keys = sorted(logs['relvqa'].keys())

    logs = {}
    for log_name in dir_logs.keys():
        logs[log_name] = load_max_logs(dir_logs[log_name], nb_epochs=args.nb_epochs)

    for key in keys:
        names = []
        values = []
        for name, log_values in logs.items():
            if key in log_values:
                names.append(name)
                values.append(log_values[key])
        values_names = sorted(zip(values,names),reverse=True)
        values_names = [[i+1, name, value] for i, (value, name) in enumerate(values_names)]
        #if values_names[0][1] == 'bottomup':   
        print('\n\n## {}\n'.format(key))     
        print(tabulate(values_names, headers=['Place', 'Method', 'Score']))
        # for i, (value, name) in enumerate(values_names):
        #     place = i+1
        #     print('{}\t{}\t{:.4f}'.format(place, name, value))


