import torch.optim.lr_scheduler
from bootstrap.lib.logger import Logger
from torch.optim.lr_scheduler import _LRScheduler

class LearningRateScheduler():

    def __init__(self, optimizer, engine=None):
        self.optimizer = optimizer
        self.scheduler = None
        self.param_groups = self.optimizer.param_groups # to keep compatibility with GradClipper
        if engine:
            engine.register_hook('train_on_start_epoch', self.step_scheduler)

    def step(self, closure=None):
        self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state = {}
        state['optimizer'] = self.optimizer.state_dict()
        state['scheduler'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])

    def step_scheduler(self):
        self.scheduler.step()
        Logger().log_value('train_epoch.lr', self.optimizer.param_groups[0]['lr'])


class StepLR(LearningRateScheduler):

    def __init__(self, optimizer, engine=None, step_size=None, gamma=0.1, last_epoch=-1):
        super(StepLR, self).__init__(optimizer, engine)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size,
            gamma=gamma,
            last_epoch=last_epoch)


class CosineAnnealingLR(LearningRateScheduler):

    def __init__(self, optimizer, engine=None, T_max=None, eta_min=0, last_epoch=-1):
        super(CosineAnnealingLR, self).__init__(optimizer, engine)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max,
            eta_min=eta_min,
            last_epoch=last_epoch)
