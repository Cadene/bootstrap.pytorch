import importlib

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from .bce import BCEWithLogitsLoss
from .cross_entropy import CrossEntropyLoss
from .nll import NLLLoss


def factory(engine=None, mode=None):
    if 'criterion' not in Options()['model'] or Options()['model']['criterion'] is None:
        return None

    Logger()(f'Creating criterion for {mode} mode...')

    if Options()['model']['criterion'].get('import', False):
        module = importlib.import_module(Options()['model']['criterion']['import'])
        criterion = module.factory(engine, mode)

    elif Options()['model']['criterion']['name'] == 'nll':
        criterion = NLLLoss(**Options()['model']['criterion'])

    elif Options()['model']['criterion']['name'] == 'cross_entropy':
        criterion = CrossEntropyLoss(**Options()['model']['criterion'])

    elif Options()['model']['criterion']['name'] == 'BCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss(**Options()['model']['criterion'])

    else:
        raise ValueError()

    return criterion
