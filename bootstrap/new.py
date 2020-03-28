from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("project_path", action="store", type=str, help="Path to new project")
parser.add_argument("project_name", type=str, help="Project name")


if __name__ == "__main__":
    args = parser.parse_args()
    path = Path(args.project_path)

    path.mkdir()
    path = path / args.project_name
    path.mkdir()

    Path(path / "models").mkdir()
    Path(path / "models/__init__.py").touch()
    Path(path / "models/networks").mkdir()
    Path(path / "models/networks/__init__.py").touch()
    Path(path / "models/networks/factory.py").touch()
    Path(path / "models/networks/mynetwork.py").touch()

    with open(path / "models/networks/factory.py", "w") as f:
        f.write(r"""from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.networks.data_parallel import DataParallel

from .mynetwork import MyNetwork


def factory(engine):
    logger = Logger()
    net_opt = Options()["model"]["network"]
    logger("Creating Network")
    if net_opt["name"] == "mynetwork":
        # You can use any param to create your network
        # You just have to write them in your option file from options/ folder
        net = MyNetwork(net_opt["param1"], net_opt["param2"])
    else:
        raise ValueError(opt["name"])
    logger("Network was created")

    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net
""")

    with open(path / "models/networks/mynetwork.py", "w") as f:
        f.write(r"""import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyNetwork, self).__init__()
        # Assign args

    def forward(self, x):
        # x is a dictionnary given by Dataset class
        pred = self.net(x)
        return pred  # This is a tensor (or several tensors)
""")
