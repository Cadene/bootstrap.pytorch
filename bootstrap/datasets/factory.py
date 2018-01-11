import importlib

from ..lib.options import Options
from ..lib.logger import Logger

def factory():
    Logger()('Creating dataset...')

    if 'import' not in Options()['dataset']:
        raise ValueError()

    # import looks like "yourmodule.datasets.factory"
    module = importlib.import_module(Options()['dataset']['import'])

    dataset = {}

    if Options()['dataset']['train_split']:
        train_split = Options()['dataset']['train_split']
        dataset['train'] = module.factory(train_split)
        Logger()('train on {}set ({} items)'.format(train_split, len(dataset['train'])))

    if Options()['dataset']['eval_split']:
        eval_split = Options()['dataset']['eval_split']
        dataset['eval'] = module.factory(eval_split)
        Logger()('evaluate on {}set ({} items)'.format(eval_split, len(dataset['eval'])))

    return dataset
