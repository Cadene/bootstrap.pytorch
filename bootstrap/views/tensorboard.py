import os
import re
import json
import math
from tensorboardX import SummaryWriter
from ..lib.logger import Logger
from ..lib.options import Options


class Tensorboard():

    def __init__(self, options):
        super(Tensorboard, self).__init__()
        self.options = options

    def generate(self):
        # erase old log files, currently there is no way to replace it nor append to it
        for filename in os.listdir(self.options['exp']['dir']):
            if 'tfevents' in filename:
                os.remove(os.path.join(self.options['exp']['dir'], filename))
        writer = SummaryWriter(log_dir=self.options['exp']['dir'])
        log_names = []
        views_per_figure = []
        # find all the log_names to load
        items = self.options['view.items'] if 'view.items' in self.options else self.options['view']
        for i, view_raw in enumerate(items):
            views = []
            for view_interim in view_raw.split('+'):
                log_name, view_name = view_interim.split(':')
                views.append({
                    'view_interim': view_interim,          # logs:train_epoch.loss
                    'log_name': log_name,                  # logs
                    'view_name': view_name,                # train_epoch.loss
                    'split_name': view_name.split('.')[0], # train_epoch
                    'log_type': view_name.split('.')[1]    # loss
                })
                log_names.append(log_name)
            views_per_figure.append(views)

        log_names = list(set(log_names)) # unique

        data_dict = {}
        for log_name in log_names:
            path_json = os.path.join(self.options['exp']['dir'], '{}.json'.format(log_name))
            if os.path.isfile(path_json):
                with open(path_json, 'r') as handle:
                    data_json = json.load(handle)        
                data_dict[log_name] = data_json
            else:
                Logger()("Json log file '{}' not found in '{}'".format(log_name, path_json), log_level=Logger.WARNING)

        nb_keys = len(items)
        
        for figure_id, views in enumerate(views_per_figure):
            for view in views:
                if view['log_name'] not in data_dict:
                    continue

                if view['view_name'] not in data_dict[view['log_name']]:
                    Logger()("View '{}' not in '{}.json'".format(view['view_name'], view['log_name']), log_level=Logger.WARNING)
                    continue

                y = data_dict[view['log_name']][view['view_name']]

                if 'epoch' in view['split_name']:
                    # example: data_dict['logs_last']['test_epoch.epoch']
                    key = view['split_name']+'.epoch' # TODO: ugly fix, to be remove
                    if key not in data_dict[view['log_name']]:
                        key = 'eval_epoch.epoch'
                    x = data_dict[view['log_name']][key]
                    subtype = 'epoch'
                else:
                    x = list(range(len(y)))
                    subtype = 'batch'

                for x_val, y_val in zip(x, y):
                    name = '{}/{}/{}'.format(view['log_type'], subtype, view['split_name'])
                    writer.add_scalar(name, y_val, x_val)

        writer.close()
        Logger()('Tensorboard view generated in '+path_json)

def view(path_opts=None):
    Tensorboard(Options(path_opts)).generate()

if __name__ == '__main__':
    from ..run import main
    main(run=view)
