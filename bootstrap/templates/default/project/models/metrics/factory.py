from bootstrap.lib.options import Options

from .mymetric import {PROJECT_NAME}Metric


def factory(engine=None, mode="train"):
    opt = Options()['model.metric']

    if opt['name'] == '{PROJECT_NAME_LOWER}metric':
        metric = {PROJECT_NAME_LOWER}metric()
    else:
        raise ValueError(opt['name'])

    return metric
