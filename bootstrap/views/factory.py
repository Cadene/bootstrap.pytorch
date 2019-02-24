import importlib
from ..lib.options import Options
from ..lib.logger import Logger

def factory(engine=None):
    Logger()('Creating views...')

    # if views not exist, pick view
    # to support backward compatibility
    if 'view' in Options():
        opt = Options()['view']
    elif 'views' in Options():
        opt = Options()['views']
    else:
        Logger()('Not a single view has been created', log_level=Logger.WARNING)
        return None

    if 'import' in opt:
        module = importlib.import_module(opt['import'])
        view = module.factory()
        return view

    exp_dir = Options()['exp.dir']

    if type(opt) == list:
        # default behavior
        view_name = 'plotly'
        items = opt
        fname = 'view.html'
    else:
        # to support backward compatibility
        view_name = opt.get('name', 'plotly')
        items = opt.get('items', None)
        fname = opt.get('file_name', 'view.html')

    view = None
    if view_name == 'tensorboard':
        # Lazy import for unused libraries
        from .tensorboard import Tensorboard
        view = Tensorboard(items, exp_dir)

    elif view_name == 'plotly':
        # Lazy import for unused libraries
        from .plotly import Plotly
        view = Plotly(items, exp_dir, fname)

    else:
        raise ValueError(view_name)

    return view
