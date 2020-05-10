from bootstrap.lib.options import Options
from .{PROJECT_NAME_LOWER} import {PROJECT_NAME}Metric  # noqa: E999


def factory(engine=None, mode='train'):
    opt = Options()['model.metric']

    if opt['name'] == '{PROJECT_NAME_LOWER}':
        metric = {PROJECT_NAME}Metric(**opt)  # noqa: E999
    else:
        raise ValueError(opt['name'])

    return metric
