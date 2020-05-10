from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from .{PROJECT_NAME_LOWER} import {PROJECT_NAME}Dataset


def factory(engine=None):
    Logger()('Creating dataset...')

    opt = Options()['dataset']

    dataset = {}

    if opt.get('train_split', None):
        dataset['train'] = factory_split(opt['train_split'])

    if opt.get('eval_split', None):
        dataset['eval'] = factory_split(opt['eval_split'])

    return dataset


def factory_split(split):
    opt = Options()['dataset']

    shuffle = ('train' in split)

    dict_opt = opt.asdict()
    dict_opt.pop('dir', None)
    dict_opt.pop('batch_size', None)
    dict_opt.pop('nb_threads', None)
    dataset = {PROJECT_NAME}Dataset(
        dir_data=opt['dir'],
        split=split,
        batch_size=opt['batch_size'],
        shuffle=shuffle,
        nb_threads=opt['nb_threads'],
        **dict_opt
    )

    return dataset
