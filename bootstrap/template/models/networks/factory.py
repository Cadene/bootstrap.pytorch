from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.networks.data_parallel import DataParallel

from .mynetwork import {PROJECT_NAME}Network


def factory(engine):
    logger = Logger()
    net_opt = Options()["model"]["network"]
    logger("Creating Network...")

    if net_opt["name"] == "{PROJECT_NAME_LOWER}network":
        # You can use any param to create your network
        # You just have to write them in your option file from options/ folder
        net = {PROJECT_NAME}Network(net_opt["param1"], net_opt["param2"])
    else:
        raise ValueError(opt["name"])
    logger("Network was created")
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)
    return net
