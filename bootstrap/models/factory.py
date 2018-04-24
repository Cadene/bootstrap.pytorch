import importlib

from ..lib.options import Options
from ..lib.logger import Logger

from .model import DefaultModel

def factory(engine=None):

    Logger()('Creating model...')

    if 'import' in Options()['model']:
        module = importlib.import_module(Options()['model']['import'])
        model = module.factory(engine)

    elif Options()['model']['name'] == 'default':
        model = DefaultModel(engine)

    else:
        raise ValueError()

    # TODO
    # if data_parallel is not None:
    #     if not cuda:
    #         raise ValueError
    #     model = nn.DataParallel(model).cuda()
    #     model.save = lambda x: x.module.save()

    if Options()['misc']['cuda']:
        model.cuda()

    return model