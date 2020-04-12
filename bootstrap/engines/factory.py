import importlib

from ..lib.logger import Logger
from ..lib.options import Options
from .engine import Engine
from .logger import LoggerEngine


def factory():
    Logger()('Creating engine...')

    if Options()['engine'].get('import', False):
        # import usually is "yourmodule.engine.factory"
        module = importlib.import_module(Options()['engine']['import'])
        engine = module.factory()

    elif Options()['engine']['name'] == 'default':
        engine = Engine()

    elif Options()['engine']['name'] == 'logger':
        engine = LoggerEngine()

    else:
        raise ValueError

    return engine
