from bootstrap.datasets.dataset import Dataset


class {PROJECT_NAME}Dataset(Dataset):
    """ Dataset of Wikipedia Comparable Article

    Parameters
    -----------
    """
    def __init__(self,
                 dir_data,
                 split='train',
                 batch_size=4,
                 shuffle=False,
                 pin_memory=False,
                 nb_threads=4):
        super({PROJECT_NAME}Dataset, self).__init__(dir_data, split, batch_size, shuffle, pin_memory, nb_threads)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
