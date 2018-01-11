import importlib

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

def factory(engine=None):

    Logger()('Creating network...')

    if 'import' in Options()['model']['network']:
        module = importlib.import_module(Options()['model']['network']['import'])
        network = module.factory(engine)

    else:
        raise ValueError()

    return network