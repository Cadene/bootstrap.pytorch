import torch.nn as nn

class BCEWithLogitsLoss(nn.Module):

    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out.squeeze(1), batch['class_id'].float().squeeze(1))
        return out