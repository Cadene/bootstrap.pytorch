import os
import sys
from bootstrap.new import new_project
from bootstrap.run import run
from tests.test_options import reset_options_instance


def test_new(tmpdir):
    new_project('MyProject', tmpdir)
    base_dir = os.path.join(tmpdir, 'myproject.bootstrap.pytorch')
    test_dir = os.path.join(base_dir, 'myproject')
    options_file = os.path.join(test_dir, 'options/myproject.yaml')
    sys.path.insert(0, base_dir)
    reset_options_instance()
    sys.argv += ['--path_opts', options_file]
    sys.argv += ['--exp.dir', 'logs/myproject/1_exp']
    sys.argv += ['--misc.cuda', 'False']
    sys.argv += ['--engine.nb_epochs', '10']
    run()

    fnames = [
        'ckpt_best_accuracy_engine.pth.tar',
        'ckpt_best_loss_optimizer.pth.tar',
        'logs.txt',
        'ckpt_best_accuracy_model.pth.tar',
        'ckpt_last_engine.pth.tar',
        'options.yaml',
        'ckpt_best_accuracy_optimizer.pth.tar',
        'ckpt_last_model.pth.tar',
        'view.html',
        'ckpt_best_loss_engine.pth.tar',
        'ckpt_last_optimizer.pth.tar',
        'ckpt_best_loss_model.pth.tar',
        'logs.json'
    ]

    for fname in fnames:
        file_path = f'logs/myproject/1_exp/{fname}'
        assert os.path.isfile(file_path)
