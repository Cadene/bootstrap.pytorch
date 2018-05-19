import collections
import torch
from torch.autograd import Variable

class Compose(object):
    """Composes several collate together.

    Args:
        transforms (list of ``Collate`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class ListDictsToDictLists(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        batch = self.ld_to_dl(batch)
        return batch

    def ld_to_dl(self, batch):
        if isinstance(batch[0], collections.Mapping):
            return {key: self.ld_to_dl([d[key] for d in batch]) for key in batch[0]}
        else:
            return batch


class PadTensors(object):

    def __init__(self, value=0, use_keys=[], avoid_keys=[]):
        self.value = value
        self.use_keys = use_keys
        if len(self.use_keys)>0:
            self.avoid_keys = []
        else:
            self.avoid_keys = avoid_keys

    def __call__(self, batch):
        batch = self.pad_tensors(batch)
        return batch

    def pad_tensors(self, batch):
        if isinstance(batch, collections.Mapping):
            out = {}
            for key, value in batch.items():
                if (key in self.use_keys) or \
                   (len(self.use_keys) == 0 and key not in self.avoid_keys):
                    out[key] = self.pad_tensors(value)
                else:
                    out[key] = value
            return out
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            max_size = [max([item.size(i) for item in batch]) for i in range(batch[0].dim())]
            max_size = torch.Size(max_size)
            n_batch = []
            for item in batch:
                if item.size() != max_size:
                    n_item = item.new(max_size).fill_(self.value)
                    # TODO: Improve this
                    if item.dim() == 1:
                        n_item[:item.size(0)] = item
                    elif item.dim() == 2:
                        n_item[:item.size(0),:item.size(1)] = item
                    elif item.dim() == 3:
                        n_item[:item.size(0),:item.size(1),:item.size(2)] = item
                    else:
                        raise ValueError
                    n_batch.append(n_item)
                else:
                    n_batch.append(item)
            return n_batch
        else:
            return batch


class StackTensors(object):

    def __init__(self, use_shared_memory=False, avoid_keys=[]):
        self.use_shared_memory = use_shared_memory
        self.avoid_keys = avoid_keys

    def __call__(self, batch):
        batch = self.stack_tensors(batch)
        return batch

    # key argument is useful for debuging
    def stack_tensors(self, batch, key=None):
        if isinstance(batch, collections.Mapping):
            out = {}
            for key, value in batch.items():
                if key not in self.avoid_keys:
                    out[key] = self.stack_tensors(value, key=key)
                else:
                    out[key] = value
            return out
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            out = None
            if self.use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            try:
                return torch.stack(batch, 0, out=out)
            except:
                import ipdb; ipdb.set_trace()
        else:
            return batch


class CatTensors(object):

    def __init__(self, use_shared_memory=False, use_keys=[], avoid_keys=[]):
        self.use_shared_memory = use_shared_memory
        self.use_keys = use_keys
        if len(self.use_keys)>0:
            self.avoid_keys = []
        else:
            self.avoid_keys = avoid_keys

    def __call__(self, batch):
        batch = self.cat_tensors(batch)
        return batch

    def cat_tensors(self, batch):
        if isinstance(batch, collections.Mapping):
            out = {}
            for key, value in batch.items():
                if (key in self.use_keys) or \
                   (len(self.use_keys) == 0 and key not in self.avoid_keys):
                    out[key] = self.cat_tensors(value)
                    if ('batch_id' not in out) and torch.is_tensor(value[0]):
                        out['batch_id'] = torch.cat([i*torch.ones(x.size(0)) \
                                                     for i,x in enumerate(value)], 0)
                else:
                    out[key] = value
            return out
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            out = None
            if self.use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            try:
                return torch.cat(batch, 0, out=out)
            except:
                import ipdb; ipdb.set_trace()
        else:
            return batch


class ToCuda(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        batch = self.to_cuda(batch)
        return batch

    def to_cuda(self, batch):
        if isinstance(batch, collections.Mapping):
            return {key: self.to_cuda(value) for key,value in batch.items()}
        elif torch.is_tensor(batch):
            # TODO: verify async usage
            return batch.cuda(async=True)
        elif type(batch).__name__ == 'Variable':
            # TODO: Really hacky
            return Variable(batch.data.cuda(async=True))
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            return [self.to_cuda(value) for value in batch]
        else:
            return batch


class ToCpu(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        batch = self.to_cpu(batch)
        return batch

    def to_cpu(self, batch):
        if isinstance(batch, collections.Mapping):
            return {key: self.to_cpu(value) for key,value in batch.items()}
        elif torch.is_tensor(batch):
            return batch.cpu()
        elif type(batch).__name__ == 'Variable':
            # TODO: Really hacky
            return Variable(batch.data.cpu())
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            return [self.to_cpu(value) for value in batch]
        else:
            return batch


class ToVariable(object):

    def __init__(self, volatile=False):
        self.volatile = volatile

    def __call__(self, batch):
        batch = self.to_variable(batch)
        return batch

    def to_variable(self, batch):
        if torch.is_tensor(batch):
            if self.volatile:
                return Variable(batch, volatile=True)
            else:
                return Variable(batch)
        elif isinstance(batch, collections.Mapping):
            return {key: self.to_variable(value) for key, value in batch.items()}
        elif isinstance(batch, collections.Sequence) and torch.is_tensor(batch[0]):
            return [self.to_variable(value) for value in batch]
        else:
            return batch


class SortByKey(object):

    def __init__(self, key='lengths', reverse=True):
        self.key = key
        self.reverse = True
        self.i = 0

    def __call__(self, batch):
        self.set_sort_keys(batch[self.key]) # must be a list
        batch = self.sort_by_key(batch)
        return batch

    def set_sort_keys(self, sort_keys):
        self.i = 0
        self.sort_keys = sort_keys

    # ugly hack to be able to sort without lambda function
    def get_key(self, _):
        key = self.sort_keys[self.i]
        self.i += 1
        if self.i >= len(self.sort_keys):
            self.i = 0
        return key

    def sort_by_key(self, batch):
        if isinstance(batch, collections.Mapping):
            return {key: self.sort_by_key(value) for key, value in batch.items()}
        elif type(batch) is list:#isinstance(batch, collections.Sequence):
            return sorted(batch, key=self.get_key, reverse=self.reverse)
        else:
            return batch
