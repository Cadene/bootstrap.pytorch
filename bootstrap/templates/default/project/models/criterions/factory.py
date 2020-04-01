from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .mycriterion import {PROJECT_NAME}Criterion


def factory(engine=None, mode=None):
    logger = Logger()
    logger('Creating criterion for {} mode...'.format(mode))

    if Options()['model']['criterion'].get('import', False):
        criterion = {PROJECT_NAME}Criterion()
    else:
        raise ValueError()

    return criterion
