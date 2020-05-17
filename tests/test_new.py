from os import path as osp
import sys
from bootstrap.new import new_project
from bootstrap.run import run
from tests.test_options import reset_options_instance


def test_new(tmpdir):
    new_project('MyProject', tmpdir)
    code_dir = osp.join(tmpdir, 'myproject.bootstrap.pytorch')

    # path needed to change import
    # https://stackoverflow.com/questions/23619595/pythons-os-chdir-function-isnt-working
    sys.path.insert(0, code_dir)

    reset_options_instance()
    sys.argv += ['--path_opts', osp.join(code_dir, 'myproject/options/myproject.yaml')]
    sys.argv += ['--exp.dir', osp.join(code_dir, 'logs/myproject/1_exp')]
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
        file_path = osp.join(code_dir, f'logs/myproject/1_exp/{fname}')
        assert osp.isfile(file_path)
