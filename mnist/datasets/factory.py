from bootstrap.lib.options import Options
from .mnist import MNIST

def factory(split):

    if Options()['dataset']['name'] == 'mnist':
        dataset = MNIST(
            dir_data=Options()['dataset']['dir'],
            split=split,
            batch_size=Options()['dataset']['batch_size'],
            nb_threads=Options()['dataset']['nb_threads'],
            pin_memory=Options()['misc']['cuda'])
    else:
        raise ValueError()

    return dataset

