import torch
import torch.nn as nn


class NLLLoss(nn.Module):

    def __init__(self,
                 target_field_name='class_id',
                 input_field_name='net_out',
                 output_field_name='loss',
                 weight=None,
                 reduction='mean',
                 ignore_index=-100,
                 **kwargs):
        super(NLLLoss, self).__init__()
        self.loss = nn.NLLLoss(weight=weight,
                               reduction=reduction,
                               ignore_index=ignore_index)

        self.target_field_name = target_field_name
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    def forward(self, net_out, batch):

        if isinstance(net_out, torch.Tensor):
            x = net_out
        else:
            x = net_out[self.input_field_name]

        out = {}
        out[self.output_field_name] = self.loss(x, batch[self.target_field_name])
        return out
