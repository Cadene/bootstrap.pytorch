from bootstrap.lib.options import Options
from .{PROJECT_NAME_LOWER} import {PROJECT_NAME}Criterion

def factory(engine=None, mode=None):
    opt = Options()['model.criterion']

    if opt['name'] == '{PROJECT_NAME_LOWER}':
        criterion = {PROJECT_NAME}Criterion(**opt)
    else:
        raise ValueError(opt['name'])

    return criterion
