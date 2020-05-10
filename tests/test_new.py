import os
import pytest

def test_new(tmpdir):
    exit_status = os.system(f'python -m bootstrap.new --project_name MyProject --project_dir {tmpdir}')
    assert exit_status == 0

    os.chdir(os.path.join(tmpdir, 'myproject.bootstrap.pytorch'))
    exit_status = os.system('python -m bootstrap.run -o myproject/options/myproject.yaml --exp.dir logs/myproject/1_exp --misc.cuda False --engine.nb_epochs 10')
    assert exit_status == 0

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
        assert os.path.isfile(f'logs/myproject/1_exp/{fname}')
