import torch.nn as nn

class NLLLoss(nn.Module):

    def __init__(self):
        super(NLLLoss, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out, batch['class_id'].squeeze(1))
        return out