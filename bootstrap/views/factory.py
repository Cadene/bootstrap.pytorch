import importlib
from ..lib.options import Options
from ..lib.logger import Logger
from .plotly import Plotly

def factory(engine=None):
    Logger()('Creating view...')

    view = Plotly
    if 'view' in Options():
	    if 'import' in Options()['view']:
	        module = importlib.import_module(Options()['engine']['import'])
	        view = module.factory()
	    else:
	    	view_name = Options().get('view.name', 'plotly')
	    	if view_name == 'tensorboard' or view_name == 'tb':
	    		# Lazy import for unused libraries
	    		from .tensorboard import Tensorboard
	    		view = Tensorboard
	    	elif view_name != 'plotly':
	    		Logger.log_message("Unknown view name '{}'. Defaulting to plotly.".format(view_name), Logger.WARNING)

    view_result = view(Options())

    return view_result
