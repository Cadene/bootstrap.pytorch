import importlib

from ..lib.options import Options
from ..lib.logger import Logger

def factory(engine=None):
    Logger()('Creating dataset...')

    if 'import' not in Options()['dataset']:
        raise ValueError()

    # import looks like "yourmodule.datasets.factory"
    module = importlib.import_module(Options()['dataset']['import'])
    dataset = module.factory(engine=engine)

    if 'train' in dataset:
        Logger()('Training will take place on {}set ({} items)'.format(dataset['train'].split, len(dataset['train'])))

    if 'eval' in dataset:
        Logger()('Evaluation will take place on {}set ({} items)'.format(dataset['eval'].split, len(dataset['eval'])))

    return dataset
