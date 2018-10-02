import torch
import torch.nn as nn


class DataParallel(nn.DataParallel):

    # def forward(self, *inputs, **kwargs):
    #     if not self.device_ids:
    #         return self.module(*inputs, **kwargs)
    #     inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    #     if len(self.device_ids) == 1:
    #         return self.module(*inputs[0], **kwargs[0])
    #     replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    #     outputs = self.parallel_apply(replicas, inputs, kwargs)
    #     return self.gather(outputs, self.output_device)

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)