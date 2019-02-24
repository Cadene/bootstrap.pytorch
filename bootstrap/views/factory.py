import importlib
from ..lib.options import Options
from ..lib.logger import Logger
from .utils import MultiViews

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

    if 'names' in opt:
        view = make_multi_views(opt, exp_dir)
        return view

    # make single view
    view = None

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

def make_multi_views(opt, exp_dir):
    views = []

    if 'tensorboard' in opt['names']:
        from .tensorboard import Tensorboard
        items = opt['items_tensorboard']
        view = Tensorboard(items, exp_dir)
        views.append(view)

    if 'plotly' in opt['names']:
        from .plotly import Plotly
        items = opt['items_plotly']
        fname = opt.get('file_name', 'view.html')
        view = Plotly(items, exp_dir, fname)
        views.append(view)

    if len(views) == 0:
        ValueError(opt['names'])

    view = MultiViews(views)
    return view
