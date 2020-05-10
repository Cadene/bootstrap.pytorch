from bootstrap.lib.options import Options
from .{PROJECT_NAME_LOWER} import {PROJECT_NAME}Metric

def factory(engine=None, mode='train'):
    opt = Options()['model.metric']

    if opt['name'] == '{PROJECT_NAME_LOWER}':
        metric = {PROJECT_NAME}Metric(**opt)
    else:
        raise ValueError(opt['name'])

    return metric
