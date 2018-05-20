import importlib
from ..lib.options import Options
from ..lib.logger import Logger
from .view import View

def factory(engine=None):
    Logger()('Creating view...')

    if 'view' in Options() and 'import' in Options()['view']:
        module = importlib.import_module(Options()['engine']['import'])
        view = module.factory()

    else:
        view = View(Options())

    return view