import importlib
from ..lib.options import Options
from ..lib.logger import Logger
from .engine import Engine
from .logger import LoggerEngine


def factory() -> Engine:
    """Using the import path from the Options, calls the engine factory from the user-defined module.
    If the engine name is default or logger (no specific import path specified),
    uses the default Engine or the LoggerEngine"""

    Logger()('Creating engine...')

    if 'import' in Options()['engine'] and Options()['engine']['import']:
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