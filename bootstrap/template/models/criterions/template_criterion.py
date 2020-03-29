import torch.nn as nn


class {PROJECT_NAME}Criterion(nn.Module):

    def __init__(self):
        super({PROJECT_NAME}Criterion, self).__init__()

    def forward(self, net_out, batch):
        # net_out : output of network
        # batch : output of dataset (after collate function)
        raise NotImplementedError
