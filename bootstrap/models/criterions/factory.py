import importlib

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .nll import NLLLoss
from .bce import BCEWithLogitsLoss
from .cross_entropy import CrossEntropyLoss
from .cross_entropy import CrossEntropyLoss

def factory(engine=None, mode=None):

    if 'criterion' not in Options()['model'] or Options()['model']['criterion'] is None:
        return None

    Logger()('Creating criterion from bootstrap for {} mode...'.format(mode))

    if Options()['model']['criterion'].get('import', False):
        module = importlib.import_module(Options()['model']['criterion']['import'])
        criterion = module.factory(engine, mode)

    elif Options()['model']['criterion']['name'] == 'nll':
        criterion = NLLLoss()

    elif Options()['model']['criterion']['name'] == 'cross_entropy':
        criterion = CrossEntropyLoss()

    elif Options()['model']['criterion']['name'] == 'BCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss()

    else:
        raise ValueError()

    return criterion