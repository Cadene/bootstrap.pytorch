import torch
import torch.nn as nn


class {PROJECT_NAME}Criterion(nn.Module):  # noqa: E999

    def __init__(self, *args, **kwargs):
        super({PROJECT_NAME}Criterion, self).__init__()  # noqa: E999
        self.bce_loss = nn.BCELoss()

    def forward(self, net_out, batch):
        pred = net_out['pred'].squeeze(1)
        target = batch['target']
        loss = self.bce_loss(pred, target)
        out = {'loss': loss}
        return out
