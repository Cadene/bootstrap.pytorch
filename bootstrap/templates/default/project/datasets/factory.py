from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .mydataset import {PROJECT_NAME}Dataset


def factory(engine=None):
    logger = Logger()
    logger('Creating dataset...')

    opt = Options()["dataset"]

    dataset = {}

    if opt.get("train_split", None):
        logger("Loading train data")
        dataset["train"] = factory_split(opt["train_split"])
        logger(f"Train dataset length is {len(dataset['train'])}")

    if opt.get("eval_split", None):
        logger("Loading test data")
        dataset["eval"] = factory_split(opt["eval_split"])
        logger(f"Test dataset length is {len(dataset['eval'])}")

    logger("Dataset was created")
    return dataset


def factory_split(split):
    opt = Options()["dataset"]

    shuffle = ("train" in split)

    dataset = {PROJECT_NAME}Dataset(
        dir_data=opt["dir"],
        split=split,
        batch_size=opt["batch_size"],
        shuffle=shuffle,
        nb_threads=opt["nb_threads"]
    )

    return dataset
