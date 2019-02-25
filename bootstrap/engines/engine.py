import os
import math
import time
import torch
import datetime
import threading
from ..lib import utils
from ..lib.options import Options
from ..lib.logger import Logger

class Engine(object):
    """Contains training and evaluation procedures
    """

    def __init__(self):
        self.hooks = {}
        self.epoch = 0
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.view = None
        self.best_out = {}

        # generate_view will be executed at the end of each
        # training and evaluation epoch
        self.register_hook('train_on_flush', self.generate_view)
        self.register_hook('eval_on_flush', self.generate_view)

    def generate_view(self):
        """ Generate a view.html via an asynchronous call to `self.view.generate()`
        """
        if self.view is not None:
            threading.Thread(target=self.view.generate).start()
        # path_opts = os.path.join(Options()['exp']['dir'], 'options.yaml')
        # os.system('python -m bootstrap.views.view --path_opts {}'.format(path_opts))

    def load_state_dict(self, state):
        """ 
        """
        self.epoch = state['epoch']
        self.best_out = state['best_out']

    def state_dict(self):
        """ 
        """
        state = {}
        state['epoch'] = self.epoch
        state['best_out'] = self.best_out
        return state

    def hook(self, name):
        """ Run all the callback functions that have been registered
            for a hook.

            Args:
                name: the name of the hook
        """
        if name in self.hooks:
            for func in self.hooks[name]:
                func()

    def register_hook(self, name, func):
        """ Register a callback function to be triggered when the hook
            is called.

            Args:
                name: the name of the hook
                func: the callback function (no argument)

            Example usage:
            
            .. code-block:: python

                def func():
                    print('hooked!')

                engine.register_hook('train_on_start_batch', func)
        """
        if name not in self.hooks:
            self.hooks[name] = []
        self.hooks[name].append(func)

    def resume(self, map_location=None):
        """ Resume a checkpoint using the `bootstrap.lib.options.Options`
        """
        Logger()('Loading {} checkpoint'.format(Options()['exp']['resume']))
        self.load(Options()['exp']['dir'],
                  Options()['exp']['resume'],
                  self.model, self.optimizer,
                  map_location=map_location)
        self.epoch += 1

    def eval(self):
        """ Launch evaluation procedures
        """
        Logger()('Launching evaluation procedures')

        if Options()['dataset']['eval_split']:
            # self.epoch-1 to be equal to the same resumed epoch
            # or to be equal to -1 when not resumed
            self.eval_epoch(self.model, self.dataset['eval'], self.epoch-1, logs_json=True)

        Logger()('Ending evaluation procedures')

    def train(self):
        """ Launch training procedures

            List of the hooks:
            
            - train_on_start: before the full training procedure

        """
        Logger()('Launching training procedures')

        self.hook('train_on_start')
        while self.epoch < Options()['engine']['nb_epochs']:
            self.train_epoch(self.model, self.dataset['train'], self.optimizer, self.epoch)

            if Options()['dataset']['eval_split']:
                out = self.eval_epoch(self.model, self.dataset['eval'], self.epoch)

                if 'saving_criteria' in Options()['engine'] and Options()['engine']['saving_criteria'] is not None:
                    for saving_criteria in Options()['engine']['saving_criteria']:
                        if self.is_best(out, saving_criteria):
                            name = saving_criteria.split(':')[0]
                            Logger()('Saving best checkpoint for strategy {}'.format(name))
                            self.save(Options()['exp']['dir'], 'best_{}'.format(name), self.model, self.optimizer)

            Logger()('Saving last checkpoint')
            self.save(Options()['exp']['dir'], 'last', self.model, self.optimizer)
            self.epoch += 1

        Logger()('Ending training procedures')

    def train_epoch(self, model, dataset, optimizer, epoch, mode='train'):
        """ Launch training procedures for one epoch

            List of the hooks:

            - train_on_start_epoch: before the training procedure for an epoch
            - train_on_start_batch: before the training precedure for a batch
            - train_on_forward: after the forward of the model
            - train_on_bachward: after the backward of the loss
            - train_on_update: after the optimization step
            - train_on_print: after the print to the terminal
            - train_on_end_batch: end of the training procedure for a batch
            - train_on_end_epoch: before saving the logs in logs.json
            - train_on_flush: end of the training procedure for an epoch
        """
        utils.set_random_seed(Options()['misc']['seed'] + epoch) # to be able to reproduce exps on reload
        Logger()('Training model on {}set for epoch {}'.format(dataset.split, epoch))
        model.train()

        timer = {
            'begin': time.time(),
            'elapsed': time.time(),
            'process': None,
            'load': None,
            'run_avg': 0
        }
        out_epoch = {}
        batch_loader = dataset.make_batch_loader()

        self.hook('train_on_start_epoch')
        for i, batch in enumerate(batch_loader):
            timer['load'] = time.time() - timer['elapsed']
            self.hook('train_on_start_batch')

            optimizer.zero_grad()
            out = model(batch)
            self.hook('train_on_forward')

            out['loss'].backward()
            #torch.cuda.synchronize()
            self.hook('train_on_backward')

            optimizer.step()
            #torch.cuda.synchronize()
            self.hook('train_on_update')

            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value('train_batch.epoch', epoch, should_print=False)
            Logger().log_value('train_batch.batch', i, should_print=False)
            Logger().log_value('train_batch.timer.process', timer['process'], should_print=False)
            Logger().log_value('train_batch.timer.load', timer['load'], should_print=False)

            for key, value in out.items():
                if torch.is_tensor(value):
                    if value.dim() <= 1:
                        value = value.item() # get number from a torch scalar
                    else:
                        continue
                if type(value) == list:
                    continue
                if type(value) == dict:
                    continue
                if key not in out_epoch:
                    out_epoch[key] = []
                out_epoch[key].append(value)
                Logger().log_value('train_batch.'+key, value, should_print=False)

            if i % Options()['engine']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{} elapsed: {} | left: {}".format(' '*len(mode),
                    datetime.timedelta(seconds=math.floor(time.time() - timer['begin'])),
                    datetime.timedelta(seconds=math.floor(timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{} process: {:.5f} | load: {:.5f}".format(' '*len(mode), timer['process'], timer['load']))
                Logger()("{} loss: {:.5f}".format(' '*len(mode), out['loss'].data.item()))
                self.hook('train_on_print')

            timer['elapsed'] = time.time()
            self.hook('train_on_end_batch')

            if Options()['engine']['debug']:
                if i > 2:
                    break

        Logger().log_value('train_epoch.epoch', epoch, should_print=True)
        for key, value in out_epoch.items():
            Logger().log_value('train_epoch.'+key, sum(value)/len(value), should_print=True)
        
        self.hook('train_on_end_epoch')
        Logger().flush()
        self.hook('train_on_flush')


    def eval_epoch(self, model, dataset, epoch, mode='eval', logs_json=True):
        """ Launch evaluation procedures for one epoch

            List of the hooks (``mode='eval'`` by default):

            - mode_on_start_epoch: before the evaluation procedure for an epoch
            - mode_on_start_batch: before the evaluation precedure for a batch
            - mode_on_forward: after the forward of the model
            - mode_on_print: after the print to the terminal
            - mode_on_end_batch: end of the evaluation procedure for a batch
            - mode_on_end_epoch: before saving the logs in logs.json
            - mode_on_flush: end of the evaluation procedure for an epoch

            Returns:
                out(dict): mean of all the scalar outputs of the model, indexed by output name, for this epoch
        """
        utils.set_random_seed(Options()['misc']['seed'] + epoch) # to be able to reproduce exps on reload
        Logger()('Evaluating model on {}set for epoch {}'.format(dataset.split, epoch))
        model.eval()

        timer = {
            'begin': time.time(),
            'elapsed': time.time(),
            'process': None,
            'load': None,
            'run_avg': 0
        }
        out_epoch = {}
        batch_loader = dataset.make_batch_loader()

        self.hook('{}_on_start_epoch'.format(mode))
        for i, batch in enumerate(batch_loader):
            timer['load'] = time.time() - timer['elapsed']
            self.hook('{}_on_start_batch'.format(mode))

            with torch.no_grad():
                out = model(batch)
            #torch.cuda.synchronize()
            self.hook('{}_on_forward'.format(mode))

            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value('{}_batch.batch'.format(mode), i, should_print=False)
            Logger().log_value('{}_batch.epoch'.format(mode), epoch, should_print=False)
            Logger().log_value('{}_batch.timer.process'.format(mode), timer['process'], should_print=False)
            Logger().log_value('{}_batch.timer.load'.format(mode), timer['load'], should_print=False)

            for key, value in out.items():
                if torch.is_tensor(value):
                    if value.dim() <= 1:
                        value = value.item() # get number from a torch scalar
                    else:
                        continue
                if type(value) == list:
                    continue
                if type(value) == dict:
                    continue
                if key not in out_epoch:
                    out_epoch[key] = []
                out_epoch[key].append(value)
                Logger().log_value('{}_batch.{}'.format(mode, key), value, should_print=False)

            if i % Options()['engine']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{}  elapsed: {} | left: {}".format(' '*len(mode), 
                    datetime.timedelta(seconds=math.floor(time.time() - timer['begin'])),
                    datetime.timedelta(seconds=math.floor(timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{}  process: {:.5f} | load: {:.5f}".format(' '*len(mode), timer['process'], timer['load']))
                self.hook('{}_on_print'.format(mode))
            
            timer['elapsed'] = time.time()
            self.hook('{}_on_end_batch'.format(mode))

            if Options()['engine']['debug']:
                if i > 10:
                    break

        out = {}
        for key, value in out_epoch.items():
            try:
                out[key] = sum(value)/len(value)
            except:
                import ipdb; ipdb.set_trace()

        Logger().log_value('{}_epoch.epoch'.format(mode), epoch, should_print=True)
        for key, value in out.items():
            Logger().log_value('{}_epoch.{}'.format(mode, key), value, should_print=True)

        self.hook('{}_on_end_epoch'.format(mode))
        if logs_json:
            Logger().flush()

        self.hook('{}_on_flush'.format(mode))
        return out

    def is_best(self, out, saving_criteria):
        """ Verify if the last model is the best for a specific saving criteria

            Args:
                out(dict): mean of all the scalar outputs of model indexed by output name
                saving_criteria(str):

            Returns:
                is_best(bool)

            Example usage:
            
            .. code-block:: python

                out = {
                    'loss': 0.2,
                    'acctop1': 87.02
                }

                engine.is_best(out, 'loss:min')
        """
        if ':min' in saving_criteria:
            name = saving_criteria.replace(':min', '')
            order = '<'
        elif ':max' in saving_criteria:
            name = saving_criteria.replace(':max', '')
            order = '>'
        else:
            error_msg = """'--engine.saving_criteria' named '{}' does not specify order,
            you need to chose between '{}' or '{}' to specify if the criteria needs to be minimize or maximize""".format(
                saving_criteria, saving_criteria+':min', saving_criteria+':max')
            raise ValueError(error_msg)
        
        if name not in out:
            raise KeyError("'--engine.saving_criteria' named '{}' not in outputs '{}'".format(name, list(out.keys())))

        if name not in self.best_out:
            self.best_out[name] = out[name]
        else:
            if eval('{} {} {}'.format(out[name], order, self.best_out[name])):
                self.best_out[name] = out[name]
                return True

        return False

    def load(self, dir_logs, name, model, optimizer, map_location=None):
        """ Load a checkpoint

            Args:
                dir_logs: directory of the checkpoint
                name: name of the checkpoint
                model: model associated to the checkpoint
                optimizer: optimizer associated to the checkpoint
        """
        path_template = os.path.join(dir_logs, 'ckpt_{}_{}.pth.tar')

        Logger()('Loading model...')
        model_state = torch.load(path_template.format(name, 'model'), map_location=map_location)
        model.load_state_dict(model_state)

        if Options()['dataset']['train_split'] is not None:
            if os.path.isfile(path_template.format(name, 'optimizer')):
                Logger()('Loading optimizer...')
                optimizer_state = torch.load(path_template.format(name, 'optimizer'), map_location=map_location)
                optimizer.load_state_dict(optimizer_state)
            else:
                Logger()('No optimizer checkpoint', log_level=Logger.WARNING)

        if os.path.isfile(path_template.format(name, 'engine')):
            Logger()('Loading engine...')
            engine_state = torch.load(path_template.format(name, 'engine'), map_location=map_location)
            self.load_state_dict(engine_state)
        else:
            Logger()('No engine checkpoint', log_level=Logger.WARNING)

    def save(self, dir_logs, name, model, optimizer):
        """ Save a checkpoint

            Args:
                dir_logs: directory of the checkpoint
                name: name of the checkpoint
                model: model associated to the checkpoint
                optimizer: optimizer associated to the checkpoint
        """
        path_template = os.path.join(dir_logs, 'ckpt_{}_{}.pth.tar')

        Logger()('Saving model...')
        model_state = model.state_dict()
        torch.save(model_state, path_template.format(name, 'model'))

        Logger()('Saving optimizer...')
        optimizer_state = optimizer.state_dict()
        torch.save(optimizer_state, path_template.format(name, 'optimizer'))

        Logger()('Saving engine...')
        engine_state = self.state_dict()
        torch.save(engine_state, path_template.format(name, 'engine'))

