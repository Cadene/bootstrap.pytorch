import os
import torch
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms

from bootstrap.datasets import transforms as bootstrap_tf

class MNIST(data.Dataset):

    def __init__(self, dir_data='data/mnist', split='train', batch_size=100, nb_threads=1, pin_memory=True):
        self.dir_data = dir_data
        self.split = split
        self.batch_size = batch_size
        self.nb_threads = nb_threads
        self.pin_memory = pin_memory

        if self.split == 'train':
            is_train = True
            self.shuffle = True
        elif self.split == 'val':
            is_train = False
            self.shuffle = False
        else:
            raise ValueError()

        self.item_tf = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])

        download = (not os.path.isdir(self.dir_data))
        self.dataset = datasets.MNIST(self.dir_data,
            train=is_train, download=download, transform=self.item_tf)

        # the usual collate function for bootstrap
        # to handle the (potentially nested) dict item format bellow
        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.StackTensors()
        ])
        
    def __getitem__(self, index):
        data, class_id = self.dataset[index]
        item = {}
        item['index'] = index
        item['data'] = data
        item['class_id'] = torch.LongTensor([class_id])
        return item

    def __len__(self):
        return len(self.dataset)

    def make_batch_loader(self):
        data_loader = data.DataLoader(self,
            batch_size=self.batch_size,
            num_workers=self.nb_threads,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False)
        return data_loader