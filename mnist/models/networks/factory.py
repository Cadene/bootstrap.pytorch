from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .net import Net

def factory(engine=None):

    Logger()('Creating mnist network...')

    if Options()['model']['network']['name'] == 'net':
        network = Net()

    else:
        raise ValueError()

    return network