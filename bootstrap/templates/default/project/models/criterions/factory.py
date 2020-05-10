from bootstrap.lib.options import Options
from .{PROJECT_NAME_LOWER} import {PROJECT_NAME}Criterion  # noqa: E999


def factory(engine=None, mode=None):
    opt = Options()['model.criterion']

    if opt['name'] == '{PROJECT_NAME_LOWER}':
        criterion = {PROJECT_NAME}Criterion(**opt)  # noqa: E999
    else:
        raise ValueError(opt['name'])

    return criterion
