import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out, batch['class_id'].view(-1))
        return out