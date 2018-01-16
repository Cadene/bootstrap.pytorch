import importlib

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .accuracy import Accuracy

def factory(engine=None, mode=None):

    if 'metric' not in Options()['model'] or Options()['model']['metric'] is None:
        return None

    Logger()('Creating metric from bootstrap for {} mode...'.format(mode))

    if 'import' in Options()['model']['metric']:
        module = importlib.import_module(Options()['model']['metric']['import'])
        metric = module.factory(engine, mode)

    elif Options()['model']['metric']['name'] == 'accuracy':
        metric = Accuracy(topk=Options()['model']['metric']['topk'])

    else:
        raise ValueError()

    return metric