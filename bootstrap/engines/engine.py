import os
import math
import time
import torch
import datetime
import threading
from ..lib import utils
from ..lib.options import Options
from ..lib.logger import Logger

class Engine():

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
        threading.Thread(target=self.view.generate).start()
        # path_opts = os.path.join(Options()['exp']['dir'], 'options.yaml')
        # os.system('python -m bootstrap.views.view --path_opts {}'.format(path_opts))

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.best_out = state['best_out']

    def state_dict(self):
        state = {}
        state['epoch'] = self.epoch
        state['best_out'] = self.best_out
        return state

    def hook(self, name):
        if name in self.hooks:
            for func in self.hooks[name]:
                func()

    def register_hook(self, name, func):
        if name not in self.hooks:
            self.hooks[name] = []
        self.hooks[name].append(func)

    def resume(self):
        Logger()('Loading {} checkpoint'.format(Options()['exp']['resume']))
        self.load(Options()['exp']['dir'],
                  Options()['exp']['resume'],
                  self.model, self.optimizer)
        self.epoch += 1

    def eval(self):
        if not Options()['dataset']['train_split']:
            Logger()('Launching evaluation procedures')

            if Options()['dataset']['eval_split']:
                # self.epoch-1 to be equal to the same resumed epoch 
                # or to be equal to -1 when not resumed
                self.eval_epoch(self.model, self.dataset['eval'], self.epoch-1, logs_json=True)

            Logger()('Ending evaluation procedures')
            os._exit(1)

    def train(self):
        Logger()('Launching training procedures')

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
                    if value.dim() == 0:
                        #value = value.detach() # not tracked by autograd anymore
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
                Logger()("{} loss: {:.5f}".format(' '*len(mode), out['loss'].data[0]))
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

        self.hook('eval_on_start_epoch')
        for i, batch in enumerate(batch_loader):
            self.hook('eval_on_start_batch')
            timer['load'] = time.time() - timer['elapsed']

            out = model(batch)
            #torch.cuda.synchronize()
            self.hook('eval_on_forward')

            timer['process'] = time.time() - timer['elapsed']
            if i == 0:
                timer['run_avg'] = timer['process']
            else:
                timer['run_avg'] = timer['run_avg'] * 0.8 + timer['process'] * 0.2

            Logger().log_value('eval_batch.batch', i, should_print=False)
            Logger().log_value('eval_batch.epoch', epoch, should_print=False)
            Logger().log_value('eval_batch.timer.process', timer['process'], should_print=False)
            Logger().log_value('eval_batch.timer.load', timer['load'], should_print=False)

            for key, value in out.items():
                if torch.is_tensor(value):
                    if value.dim() == 0:
                        #value = value.detach() # not tracked by autograd anymore
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
                Logger().log_value('eval_batch.'+key, value, should_print=False)

            if i % Options()['engine']['print_freq'] == 0:
                Logger()("{}: epoch {} | batch {}/{}".format(mode, epoch, i, len(batch_loader) - 1))
                Logger()("{}  elapsed: {} | left: {}".format(' '*len(mode), 
                    datetime.timedelta(seconds=math.floor(time.time() - timer['begin'])),
                    datetime.timedelta(seconds=math.floor(timer['run_avg'] * (len(batch_loader) - 1 - i)))))
                Logger()("{}  process: {:.5f} | load: {:.5f}".format(' '*len(mode), timer['process'], timer['load']))
                self.hook('eval_on_print')
            
            timer['elapsed'] = time.time()
            self.hook('eval_on_end_batch')

            if Options()['engine']['debug']:
                if i > 10:
                    break

        out = {}
        for key, value in out_epoch.items():
            try:
                out[key] = sum(value)/len(value)
            except:
                import ipdb; ipdb.set_trace()

        Logger().log_value('eval_epoch.epoch', epoch, should_print=True)
        for key, value in out.items():
            Logger().log_value('eval_epoch.'+key, value, should_print=True)

        self.hook('eval_on_end_epoch')
        if logs_json:
            Logger().flush()

        self.hook('eval_on_flush')
        return out

    def is_best(self, out, saving_criteria):
        if ':lt' in saving_criteria:
            name = saving_criteria.replace(':lt', '') # less than
            order = '<' #first_best = float('+inf')
        elif ':gt' in saving_criteria:
            name = saving_criteria.replace(':gt', '') # greater than
            order = '>' #first_best = float('-inf')
        else:
            error_msg = """'--engine.saving_criteria' named '{}' does not specify order
            by ending with a ':lt' (new best value < last best value) or 
            ':gt' (resp. >) to specify order""".format(saving_criteria)
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

    def load(self, dir_logs, name, model, optimizer):
        path_template = os.path.join(dir_logs, 'ckpt_{}_{}.pth.tar')

        Logger()('Loading model...')
        model_state = torch.load(path_template.format(name, 'model'))
        model.load_state_dict(model_state)

        if Options()['dataset']['train_split'] is not None:
            Logger()('Loading optimizer...')
            optimizer_state = torch.load(path_template.format(name, 'optimizer'))
            optimizer.load_state_dict(optimizer_state)

        Logger()('Loading engine...')
        engine_state = torch.load(path_template.format(name, 'engine'))
        self.load_state_dict(engine_state)

    def save(self, dir_logs, name, model, optimizer):
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

