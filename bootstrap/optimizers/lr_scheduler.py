import torch.optim.lr_scheduler
from bootstrap.lib.logger import Logger

class LearningRateScheduler():

    def __init__(self, optimizer, name='StepLR', step_size=1, gamma=1):
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups # to keep compatibility with GradClipper
        self.scheduler = torch.optim.lr_scheduler.__dict__[name](
            optimizer,
            step_size,
            gamma=gamma)

    def step(self):
        self.scheduler.step()
        self.optimizer.step()
        Logger().log_value('train_batch.lr',self.optimizer.param_groups[0]['lr'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)