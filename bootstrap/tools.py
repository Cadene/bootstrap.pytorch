import os
import sys
import torch
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import run


def reset_instance():
    Options._Options__instance = None
    Options.__instance = None
    Logger._Loger_instance = None
    Logger.perf_memory = {}
    sys.argv = [sys.argv[0]]  # reset command line args


def get_engine(
    path_experiment, weights="best_eval_epoch.accuracy_top1", logs_name="tools",
):
    reset_instance()
    path_yaml = os.path.join(path_experiment, "options.yaml")
    opt = Options(path_yaml)
    if weights is not None:
        opt["exp.resume"] = weights
    opt["exp.dir"] = path_experiment
    opt["misc.logs_name"] = logs_name
    engine = run(train_engine=False, eval_engine=False)
    return engine


def item_to_batch(engine, split, item, prepare_batch=True):
    batch = engine.dataset[split].collate_fn([item])
    if prepare_batch:
        batch = engine.model.prepare_batch(batch)
    return batch


def apply_item(engine, item, split="eval"):
    # item = engine.dataset[split][idx]
    engine.model.eval()
    batch = item_to_batch(engine, split, item)
    with torch.no_grad():
        out = engine.model.network(batch)
    return out


def load_model_state(engine, path):
    """
    engine: bootstran Engine
    path: path to model weights
    """
    model_state = torch.load(path)
    engine.model.load_state_dict(model_state)


def load_epoch(
    engine, epoch, exp_dir,
):
    path = os.path.join(exp_dir, f"ckpt_epoch_{epoch}_model.pth.tar")
    print(path)
    load_model_state(engine, path)


def load_last(engine, exp_dir):
    path = os.path.join(exp_dir, "ckpt_last_model.pth.tar")
    load_model_state(engine, path)

