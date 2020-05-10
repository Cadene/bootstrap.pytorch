import torch
import torch.utils.data as tdata
from bootstrap.datasets import transforms as btf


class {PROJECT_NAME}Dataset(tdata.Dataset):

    def __init__(self,
            dir_data,
            split='train',
            batch_size=4,
            shuffle=False,
            pin_memory=False,
            nb_threads=4,
            *args,
            **kwargs):
        self.dir_data = dir_data
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.nb_threads = nb_threads
        self.sampler = None

        self.collate_fn = btf.Compose([
            btf.ListDictsToDictLists(),
            btf.StackTensors()
        ])

        self.nb_items = kwargs['nb_items']
        self.data = torch.randn(self.nb_items, 10)
        self.target = torch.zeros(self.nb_items)
        self.target[:int(self.nb_items / 2)].fill_(1)
        #self.target[int(self.nb_items / 2):, 0].fill_(1)

    def make_batch_loader(self, batch_size=None, shuffle=None):
        batch_loader = tdata.DataLoader(
            dataset=self,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=self.shuffle if shuffle is None else shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.nb_threads,
            collate_fn=self.collate_fn,
            sampler=self.sampler)
        return batch_loader

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        item['data'] = self.data[idx]
        item['target'] = self.target[idx]
        return item
