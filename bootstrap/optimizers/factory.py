import copy
import torch
import importlib
from ..lib.options import Options
from ..lib.logger import Logger

from . import lr_scheduler
from .grad_clipper import GradClipper

def factory(model, engine=None):
    if not 'optimizer' in Options():
        return None

    if Options()['optimizer'].get('import', False):
        # import usually is "yourmodule.optimizers.factory"
        module = importlib.import_module(Options()['optimizer']['import'])
        optimizer = module.factory(model, engine)
    else:
        optimizer = factory_optimizer(model)

        if 'lr_scheduler' in Options()['optimizer']:
            optimizer = factory_scheduler(optimizer, engine)

        if 'grad_clip' in Options()['optimizer']:
            optimizer = factory_grad_clip(optimizer)

    return optimizer


def factory_optimizer(model):
    Logger()('Creating optimizer {} ...'.format(Options()['optimizer']['name']))

    weight_decay = Options()['optimizer'].get('weight_decay', 0)

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

    return optimizer


def factory_scheduler(optimizer, engine=None):
    Logger()('Creating lr_scheduler {}...'.format(Options()['optimizer']['lr_scheduler']['name']))
    opt = copy.copy(Options()['optimizer']['lr_scheduler'])
    optimizer = lr_scheduler.__dict__[opt.pop('name', None)](optimizer, engine, **opt)
    return optimizer


def factory_grad_clip(optimizer):
    Logger()('Creating grad_clipper {}...'.format(Options()['optimizer']['grad_clip']))
    if Options()['optimizer']['grad_clip'] > 0:
        optimizer = GradClipper(optimizer,
            grad_clip=Options()['optimizer']['grad_clip'])
    return optimizer




