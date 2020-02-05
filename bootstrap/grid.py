import ray
from ray import tune
import os
import argparse
import yaml

from bootstrap.run import run


def train_func(config):
    # change exp dir

    option_path = config.pop("option_file")
    os.chdir(config.pop("run_dir"))
    exp_dir = config.pop("exp_dir_prefix")

    override_options = {
        "resume": "last",
    }

    for name, value in config.items():
        override_options[name] = value
        if type(value) == list:
            value_str = ",".join(str(x) for x in value)
        else:
            value_str = str(value)
        exp_dir += f"--{name.split('.')[-1]}_{value_str}"

    override_options["exp.dir"] = exp_dir
    run(path_opts=option_path, override_options=override_options, run_parser=False)


def build_tune_config(option_path):
    with open(option_path, "r") as yaml_file:
        options = yaml.load(yaml_file)
    config = {}
    for key, values in options["gridsearch"].items():
        config[key] = tune.grid_search(values)
    config["exp_dir_prefix"] = options["exp"]["dir"]
    config["option_file"] = option_path
    config["run_dir"] = os.getcwd()
    return config, config["exp_dir_prefix"]


def grid(path_opts, cpu_per_trial=2, gpu_per_trial=0.5):
    config, name = build_tune_config(path_opts)
    ray.init()
    tune.run(
        train_func,
        name=name,
        # stop={"avg_inc_acc": 100},
        config=config,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        local_dir="ray_results",
    )

    # TODO: tune analysis to get best results.
    # For this, we need to extract the best score for each experiment.
    # analysis = tune.run(
    #     train_mnist, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})
    # print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--path_opts", required=True, help="Main file")
    parser.add_argument(
        "-g",
        "--gpu",
        type=float,
        default=0.5,
        help="Percentage of gpu needed for one training",
    )
    parser.add_argument(
        "-c",
        "--cpu",
        type=float,
        default=2,
        help="Percentage of gpu needed for one training",
    )
    args = parser.parse_args()
    grid(args.path_opts, args.cpu, args.gpu)


if __name__ == "__main__":
    main()
