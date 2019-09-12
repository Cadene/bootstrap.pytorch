import torch
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self,
                 topk=None,
                 target_field_name='class_id',
                 input_field_name='net_out',
                 output_field_name='accuracy_top',
                 **kwargs):
        super(Accuracy, self).__init__()

        if topk is None:
            topk = [1, 5]
        self.topk = topk

        self.target_field_name = target_field_name
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    def __call__(self, cri_out, net_out, batch):

        if isinstance(net_out, torch.Tensor):
            x = net_out
        else:
            x = net_out[self.input_field_name]

        target = batch[self.target_field_name]

        acc_out = accuracy(x.detach().cpu(),
                           target.detach().cpu(),
                           topk=self.topk)

        out = {}
        for i, k in enumerate(self.topk):
            out[f'{self.output_field_name}{k}'] = acc_out[i]
        return out


def accuracy(output, target, topk, ignore_index=None):
    """Computes the precision@k for the specified values of k"""

    if ignore_index is not None:
        target_mask = (target != ignore_index)
        target = target[target_mask]
        output_mask = target_mask.unsqueeze(1)
        output_mask = output_mask.expand_as(output)
        output = output[output_mask]
        output = output.view(-1, output_mask.size(1))

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size)[0])
    return res
