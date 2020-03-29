import torch.nn as nn


class {PROJECT_NAME}Network(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyNetwork, self).__init__()
        # Assign args

    def forward(self, x):
        # x is a dictionnary given by Dataset class
        pred = self.net(x)
        return pred  # This is a tensor (or several tensors)