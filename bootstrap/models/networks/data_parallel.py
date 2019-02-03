import torch
import torch.nn as nn
from torch.nn.parallel._functions import Gather

def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if torch.is_tensor(out):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class DataParallel(nn.DataParallel):

    def __getattr__(self, key):
        try:
            return super(DataParallel, self).__getattr__(key)
        except AttributeError:
            return self.module.__getattribute__(key)

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __getattr__(self, key):
        try:
            return super(DistributedDataParallel, self).__getattr__(key)
        except AttributeError:
            return self.module.__getattribute__(key)

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)
