import importlib

from ..lib.logger import Logger
from ..lib.options import Options
from .model import DefaultModel, SimpleModel


def factory(engine=None):

    Logger()('Creating model...')

    if Options()['model'].get('import', False):
        module = importlib.import_module(Options()['model']['import'])
        model = module.factory(engine)

    elif Options()['model']['name'] == 'default':
        model = DefaultModel(engine)

    elif Options()['model']['name'] == 'simple':
        model = SimpleModel(engine)

    else:
        raise ValueError()

    # TODO
    # if data_parallel is not None:
    #     if not cuda:
    #         raise ValueError
    #     model = nn.DataParallel(model).cuda()
    #     model.save = lambda x: x.module.save()

    if Options()['misc']['cuda']:
        Logger()('Enabling CUDA mode...')
        model.cuda()

    return model
