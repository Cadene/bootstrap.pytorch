import importlib
import torch
import torch.optim.lr_scheduler
from ..lib.options import Options
from ..lib.logger import Logger

def factory(model, engine):
    Logger()('Creating optimizer...')

    if 'import' in Options()['optimizer']:
        # import usually is "yourmodule.optimizers.factory"
        module = importlib.import_module(Options()['optimizer']['import'])
        optimizer = module.factory(model, engine)

    else:

        if 'weight_decay' in Options()['optimizer']:
            weight_decay = Options()['optimizer']['weight_decay']
        else:
            weight_decay = 0

        if Options()['optimizer']['name'] == 'adam':
            optimizer = torch.optim.Adam(
                # optimize only when parameters have requires_grad=True
                # useful to avoid optimizing nn.Embedding in a NLP setup
                filter(lambda p: p.requires_grad, model.network.parameters()),
                Options()['optimizer']['lr'],
                weight_decay=weight_decay)

        elif Options()['optimizer']['name'] == 'sgd':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.network.parameters()),
                lr=Options()['optimizer']['lr'],
                momentum=Options()['optimizer']['momentum'],
                weight_decay=weight_decay)

        else:
            raise ValueError()

    if 'lr_scheduler' in Options()['optimizer']:


        optimizer = wrap_lr_scheduler(optimizer,
            name=Options()['optimizer']['lr_scheduler']['name'],
            step_size=Options()['optimizer']['lr_scheduler']['step_size'],
            gamma=Options()['optimizer']['lr_scheduler']['gamma'])

    return optimizer


def wrap_lr_scheduler(optimizer, name='StepLR', step_size=1, gamma=1):
    Logger()('Creating scheduler {}...'.format(name))

    optimizer = torch.optim.lr_scheduler.__dict__[name](
        optimizer,
        step_size,
        gamma=gamma)

    # ugly hack to add zero_grad method to wrapper optimizer
    optimizer.step_scheduler = optimizer.step
    def step(self):
        self.step_scheduler()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    setattr(optimizer.__class__, 'step', step)
    setattr(optimizer.__class__, 'zero_grad', zero_grad)
    setattr(optimizer.__class__, 'state_dict', state_dict)
    setattr(optimizer.__class__, 'load_state_dict', load_state_dict)
    return optimizer