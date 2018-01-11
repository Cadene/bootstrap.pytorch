import torch.nn as nn

class Accuracy(nn.Module):

    def __init__(self, topk=[1,5]):
        super(Accuracy, self).__init__()
        self.topk = topk

    def __call__(self, cri_out, net_out, batch):
        out = {}
        acc_out = accuracy(net_out.data.cpu(),
                           batch['class_id'].data.cpu(),
                           topk=self.topk)
        for i, k in enumerate(self.topk):
            out['accuracy_top{}'.format(k)] = acc_out[i]
        return out

def accuracy(output, target, topk=[1,5], ignore_index=None):
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