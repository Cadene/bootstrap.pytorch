from os import path as osp
import os
import shutil
import sys
from bootstrap.new import new_project
from tests.test_options import reset_options_instance
from bootstrap.grid import main as main_grid


def test_grid(tmpdir):
    new_project("MyProject", tmpdir)
    code_dir = osp.join(tmpdir, "myproject.bootstrap.pytorch")
    path_opts = osp.join(code_dir, "myproject/options/options-grid.yaml")
    shutil.copy("tests/options-grid.yaml", path_opts)
    os.chdir(code_dir)

    expected_exp_dirs = [
        "logs/myproject/1_exp--lr_0.1--seed_1337",
        "logs/myproject/1_exp--lr_0.1--seed_42",
        "logs/myproject/1_exp--lr_0.001--seed_1337",
        "logs/myproject/1_exp--lr_0.001--seed_42",
    ]

    # path needed to change import
    # https://stackoverflow.com/questions/23619595/pythons-os-chdir-function-isnt-working
    sys.path.insert(0, code_dir)
    reset_options_instance()
    sys.argv += ["--path_opts", path_opts]
    sys.argv += ["--gpu-per-trial", "0.0"]
    sys.argv += ["--cpu-per-trial", "0.5"]
    main_grid()

    fnames = [
        "ckpt_best_accuracy_engine.pth.tar",
        "ckpt_best_loss_optimizer.pth.tar",
        "logs.txt",
        "ckpt_best_accuracy_model.pth.tar",
        "ckpt_last_engine.pth.tar",
        "options.yaml",
        "ckpt_best_accuracy_optimizer.pth.tar",
        "ckpt_last_model.pth.tar",
        "view.html",
        "ckpt_best_loss_engine.pth.tar",
        "ckpt_last_optimizer.pth.tar",
        "ckpt_best_loss_model.pth.tar",
        "logs.json",
    ]
    for exp_dir in expected_exp_dirs:
        for fname in fnames:
            file_path = osp.join(code_dir, f"{exp_dir}/{fname}")
            assert osp.isfile(file_path)
