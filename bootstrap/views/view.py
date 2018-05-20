import os
import re
import json
import math
import argparse
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, plot
#from threading import Thread
from ..lib.logger import Logger
from ..lib.options import Options

def seaborn_color_to_plotly(list_color):
    n_list_color = []
    for color in list_color:
        n_color = []
        for value in list(color):
            n_color.append(str(int(value * 255)))
        n_list_color.append('rgb('+','.join(n_color)+')')
    return n_list_color


# class ViewThread(Thread):

#     def __init__(self, view):
#         super(View, self).__init__()
#         self.view

#     def run(self):
#         self.view.generate_view


class View():

    def __init__(self, options):
        super(View, self).__init__()
        self.options = options

    # def start_thread(self):
    #     thread = ViewThread(self)
    #     thread.start()
        
    def generate(self):
        # find all the log_names to load
        log_names = []
        views_per_figure = []
        for i, view_raw in enumerate(self.options['view']):
            views = []
            for view_interim in view_raw.split('+'):
                log_name, view_name = view_interim.split(':')
                views.append({
                    'view_interim': view_interim,         # logs:train_epoch.loss
                    'log_name': log_name,                 # logs
                    'view_name': view_name,               # train_epoch.loss
                    'split_name': view_name.split('.')[0] # train_epoch
                })
                log_names.append(log_name)
            views_per_figure.append(views)

        log_names = list(set(log_names)) # unique

        data_dict = {}
        for log_name in log_names:
            path_json = os.path.join(self.options['exp']['dir'],
                                     '{}.json'.format(log_name))
            if os.path.isfile(path_json):
                with open(path_json, 'r') as handle:
                    data_json = json.load(handle)        
                data_dict[log_name] = data_json
            else:
                Logger()("Json log file '{}' not found in '{}'".format(log_name, path_json), log_level=Logger.WARNING)

        nb_keys = len(self.options['view'])
        nb_rows = math.ceil(nb_keys / 2)
        nb_cols = min(2, nb_keys)

        figure = tools.make_subplots(rows=nb_rows, cols=nb_cols,
            subplot_titles=self.options['view'],
            print_grid=False)

        colors = {'train_epoch': 'rgb(214, 39, 40)', 'train_batch': 'rgb(214, 39, 40)',
                  #'trainval_epoch': 'rgb(214, 39, 40)', 'trainval_batch': 'rgb(214, 39, 40)',
                  'val_epoch': 'rgb(31, 119, 180)', 'val_batch': 'rgb(31, 119, 180)',
                  'eval_epoch': 'rgb(31, 119, 180)', 'eval_batch': 'rgb(31, 119, 180)',
                  'test_epoch': 'rgb(31, 180, 80)', 'test_batch': 'rgb(31, 180, 80)'}

        for figure_id, views in enumerate(views_per_figure):
            
            figure_pos_x = figure_id % 2 + 1
            figure_pos_y = int(figure_id/2) + 1

            for view in views:
                if view['log_name'] not in data_dict:
                    continue

                if view['view_name'] not in data_dict[view['log_name']]:
                    Logger()("View '{}' not in '{}.json'".format(view['view_name'], view['log_name']), log_level=Logger.WARNING)
                    continue

                if view['split_name'] not in colors:
                    Logger()("Split '{}' not in colors '{}'".format(view['split_name'], list(colors.keys())), log_level=Logger.WARNING)
                    color = colors['train_epoch']
                else:
                    color = colors[view['split_name']]

                y = data_dict[view['log_name']][view['view_name']]

                if 'epoch' in view['split_name']:
                    # example: data_dict['logs_last']['test_epoch.epoch']
                    key = view['split_name']+'.epoch' # TODO: ugly fix, to be remove
                    if key not in data_dict[view['log_name']]:
                        key = 'eval_epoch.epoch'
                    x = data_dict[view['log_name']][key]
                else:
                    x = list(range(len(y)))

                scatter = go.Scatter(
                    x = x,
                    y = y,
                    name = view['view_interim'],
                    line = dict(color=color)
                )
                figure.append_trace(scatter, figure_pos_y, figure_pos_x)

        figure['layout'].update(
            autosize=False,
            width=1800,
            height=400*nb_rows
        )
        path_view = os.path.join(self.options['exp']['dir'], 'view.html')
        plot(figure, filename=path_view, auto_open=False)
        Logger()('View generated in '+path_view)


# def generate_multi_view():
#     nb_keys = len(Options()['logs']['views'])
#     nb_rows = math.ceil(nb_keys / 2)
#     nb_cols = min(2, nb_keys)

#     nb_exps = len(Options()['exp']['dirs'])
#     colors = [seaborn_color_to_plotly(sns.hls_palette(nb_exps, l=.30)), # dark for train
#               seaborn_color_to_plotly(sns.hls_palette(nb_exps)),        # medium for val
#               seaborn_color_to_plotly(sns.hls_palette(nb_exps, l=.70))] # light for test

#     figure = tools.make_subplots(rows=nb_rows, cols=nb_cols,
#             subplot_titles=Options()['logs']['views'],
#             print_grid=False)

#     path_view = os.path.join(os.path.dirname(Options()['exp']['dirs'][0]), 'view.html')

#     for exp_id, exp_dir in enumerate(Options()['exp']['dirs']):    
#         path_logs = os.path.join(exp_dir, 'logs.json')
#         exp_name = os.path.basename(exp_dir.rstrip('/'))

#         with open(path_logs, 'r') as handle:
#             data_json = json.load(handle)

#         for s_id, split in enumerate(Options()['logs']['views']['splits']):
#             df = pandas.DataFrame(data_json[split])
#             color = colors[s_id][exp_id]

#             for i, key in enumerate(Options()['logs']['views']):
#                 x = i % 2 + 1
#                 y = int(i/2) + 1
#                 if 'groupby' in Options()['logs']['views']:
#                     groupby = Options()['logs']['views']['groupby']
#                     data = list(df[[groupby, key]].groupby([groupby])[key].mean())
#                 else:
#                     data = list(df[key])
#                 figure.append_trace(go.Scatter(
#                     x = list(range(len(data))),
#                     y = data,
#                     name = split+'_'+key+'_'+exp_name,
#                     line = dict(color=color)
#                 ), y, x)
#     figure['layout'].update(
#         autosize=False,
#         width=1800,
#         height=400*nb_rows
#     )
#     plot(figure, filename=path_view, auto_open=False)
#     print('View generated in '+path_view)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_opts', default='', type=str)
    args = parser.parse_args()

    view = View(Options(args.path_opts))
    view.generate()
    # # Load options.yaml in exp.dir
    # if 'dirs' in Options()['exp'] and len(Options()['exp']['dirs']) > 0:
    #     generate_multi_view()
    # elif 'dir' in Options()['exp']:
    #     generate_view()
    # else:
    #     raise ValueError

if __name__ == '__main__':
    main()
