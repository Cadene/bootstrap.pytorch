import torch
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from .{PROJECT_NAME_LOWER} import {PROJECT_NAME}Network

def factory(engine):
    opt = Options()['model.network']

    if opt['name'] == '{PROJECT_NAME_LOWER}':
       net = {PROJECT_NAME}Network(**opt)
    else:
        raise ValueError(opt['name'])

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)
    return net
