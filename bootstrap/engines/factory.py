import importlib

from .engine import Engine
from .logger import LoggerEngine
from ..lib.logger import Logger
from ..lib.options import Options


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
