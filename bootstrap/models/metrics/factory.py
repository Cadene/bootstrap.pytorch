import importlib

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .accuracy import Accuracy

def factory(engine=None, mode=None):

    Logger()('Creating metric...')

    if 'import' in Options()['model']['metric']:
        module = importlib.import_module(Options()['model']['metric']['import'])
        metric = module.factory(engine, mode)

    elif Options()['model']['metric']['name'] == 'accuracy':
        metric = Accuracy(topk=Options()['model']['metric']['topk'])

    else:
        raise ValueError()

    return metric