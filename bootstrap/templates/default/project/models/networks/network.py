import torch.nn as nn


class {PROJECT_NAME}Network(nn.Module):  # noqa: E999

    def __init__(self, *args, **kwargs):
        super({PROJECT_NAME}Network, self).__init__()  # noqa: E999
        self.net = nn.Sequential(
            nn.Linear(kwargs['dim_in'], kwargs['dim_out']),
            nn.Sigmoid())

    def forward(self, batch):
        x = batch['data']
        y = self.net(x)
        out = {'pred': y}
        return out
