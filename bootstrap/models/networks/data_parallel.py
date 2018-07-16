import torch
import torch.nn as nn

class DataParallel(nn.DataParallel):

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)