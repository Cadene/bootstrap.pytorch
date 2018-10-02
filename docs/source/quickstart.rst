Quickstart
==========

Starting a project
------------------

After installing pytorch, we need to clone bootstrap.pytorch. In this example, we will build our new project upon the existing `mnist.bootstrap.pytorch <https://github.com/Cadene/mnist.bootstrap.pytorch>`_ submodule. 

.. code-block:: bash

    git clone https://github.com/Cadene/bootstrap.pytorch.git
    cd bootstrap.pytorch
    git submodule update --init mnist


Running an experiment
---------------------

To display parsed options from the yaml file:

.. code-block:: bash
    
    python -m bootstrap.run
           -o mnist/options/sgd.yaml
           -h

To run an experiment (training + evaluation):

.. code-block:: bash

    python -m bootstrap.run
           -o mnist/options/sgd.yaml

You can also overwrite default options using the command line:

.. code-block:: bash

    python -m bootstrap.run
           -o mnist/options/sgd.yaml
           --exp.dir logs/mnist/my_sgd


Running an experiment will create 4 files in your experiment directory (logs/mnist/my_sgd):

- `options.yaml <https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/mnist/sgd/options.yaml>`_ contains the options used for the experiment;
- `logs.txt <https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/mnist/sgd/logs.txt>`_ contains all the information printed by the logger;
- `logs.json <https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/mnist/sgd/logs.json>`_ contains trainin data logs, such as: train_epoch.loss, train_batch.loss, eval_epoch.accuracy_top1, etc;
- `view.html <https://rawgit.com/Cadene/bootstrap.pytorch/master/logs/mnist/sgd/view.html>`_ contains training and evaluation curves with javascript utilities (plotly).

Depending on your options, some checkpoints may also be created:

- ckpt_last_engine.pth.tar
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar
- ckpt_best_acctop1_engine.pth.tar
- ckpt_best_acctop1_model.pth.tar
- ckpt_best_acctop1_optimizer.pth.tar

Then, a model can be easily initialized from a checkpoint in order to resume training. You can also choose which checkpoint to initialize it from:

.. code-block:: bash

    python -m bootstrap.run
           -o logs/mnist/my_sgd/options.yaml
           --exp.resume last

Evaluating a trained model is also simple. This time, let's suppose you want to evaluate your model on the test set, but loading the checkpoint with the best top-1 accuracy on the validation set. The command then becomes:

.. code-block:: bash

    python -m bootstrap.run
           -o logs/mnist/my_sgd/options.yaml
           --exp.resume best_acctop1
           --dataset.train_split
           --dataset.eval_split test


Adding a custom network
------------------------

Create a new torch.nn.Module in mnist/models/networks/my_network.py

Add it in the factory in mnist/models/networks/factory.py

Create a new options for it:

.. code-block:: bash

    cp mnist/options/default.yaml mnist/options/my_options.yaml


Adding a custom criterion
-------------------------


Adding a custom metric
----------------------


Adding a custom dataset
-----------------------


Adding a custom workflow
------------------------

.. code-block:: bash

    python -m mnist.custom_run
           -o logs/mnist/my_sgd/options.yaml
           --exp.resume best_acctop1
           --dataset.train_split
           --dataset.eval_split test


