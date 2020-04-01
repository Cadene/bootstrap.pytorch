import torch.nn as nn
from bootstrap.lib.logger import Logger


class {PROJECT_NAME}Metric(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, crit_out, net_out, batch):
        # crit_out : output of criterion (dictionnary)
        # net_out : output of network
        # batch : output of dataset (after collate function)
        raise NotImplementedError
