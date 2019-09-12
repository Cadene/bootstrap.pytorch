import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    """ Defines a BCE with logits loss.

        Args:
            input_field_name (string): Specifies the field name of the input if net_out is a `dict`.
            target_field_name (string): Specifies the field name of the target.
            output_field_name (string): Specifies the field name of the output.

        Returns:
            dict: Returns a dict with the loss value.

    """

    def __init__(self,
                 target_field_name='class_id',
                 input_field_name='net_out',
                 output_field_name='loss',
                 weight=None,
                 reduction='mean',
                 pos_weight=None,
                 **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight=weight,
                                         reduction=reduction,
                                         pos_weight=pos_weight)

        self.target_field_name = target_field_name
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    def forward(self, net_out, batch):

        if isinstance(net_out, torch.Tensor):
            x = net_out
        else:
            x = net_out[self.input_field_name]

        out = {}
        out[self.output_field_name] = self.loss(x, batch[self.target_field_name].float())
        return out
