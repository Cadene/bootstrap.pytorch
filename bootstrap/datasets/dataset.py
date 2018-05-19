import torch.utils.data as data
import itertools
from . import transforms as bootstrap_tf

class Dataset(data.Dataset):

    def __init__(self, dir_data,
                 split='train',
                 batch_size=4,
                 shuffle=False,
                 pin_memory=False,
                 nb_threads=4
                 ):
        self.dir_data = dir_data
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.nb_threads = nb_threads

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.StackTensors()
        ])

    def make_batch_loader(self):
        batch_loader = data.DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.nb_threads,
            collate_fn=self.collate_fn)
        return batch_loader


class ListDatasets(data.Dataset):

    def __init__(self, datasets,
                 split='train',
                 batch_size=4,
                 shuffle=False,
                 pin_memory=False,
                 nb_threads=4,
                 bootstrapping=False,
                 seed=1337):
        self.datasets = datasets
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.nb_threads = nb_threads
        self.bootstrapping = bootstrapping
        self.seed = seed

        self.make_lengths_and_ids()

    def make_lengths_and_ids(self):
        self.lengths = [len(d) for d in self.datasets]
        self.cum_lengths = list(itertools.accumulate(self.lengths))
        self.cum_lengths_min = [0] + self.cum_lengths

        self.collate_fn = self.datasets[0].collate_fn

        if self.bootstrapping:
            self.ids = self.make_bootstrapping()
        else:
            self.ids = range(sum(self.lengths))

    def make_bootstrapping(self):
        nb_items = sum(self.lengths)
        rnd = np.random.RandomState(seed=self.seed)
        indices = rnd.choice(nb_items,
                             size=int(nb_items*0.95),
                             replace=False)
        if self.split != 'train':
            indices = np.array(list(set(np.arange(nb_items)) - set(indices)))
        return indices

    def make_batch_loader(self):
        batch_loader = data.DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.nb_threads,
            collate_fn=self.collate_fn)
        return batch_loader

    def __getitem__(self, index):
        idx = self.ids[index]
        item = None
        for i, cum_len in enumerate(self.cum_lengths):
            if idx < cum_len:
                item = self.datasets[i][idx - self.cum_lengths_min[i]]
                break
        return item

    def __len__(self):
        return len(self.ids)
