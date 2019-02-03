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

- [options.yaml](https://github.com/Cadene/bootstrap.pytorch/blob/master/docs/assets/logs/mnist/sgd/options.yaml) contains the options used for the experiment,
- [logs.txt](https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/docs/assets/logs/mnist/sgd/logs.txt) contains all the information given to the logger.
- [logs.json](https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/docs/assets/logs/mnist/sgd/logs.json) contains the following data: train_epoch.loss, train_batch.loss, eval_epoch.accuracy_top1, etc.
- <a href="http://htmlpreview.github.io/?https://raw.githubusercontent.com/Cadene/bootstrap.pytorch/master/docs/assets/logs/mnist/sgd/view.html">view.html</a> contains training and evaluation curves with javascript utilities (plotly).

Depending on your options, some checkpoints may also be created:

.. code-block:: bash

    ls logs/mnist/my_sgd
    > ckpt_last_engine.pth.tar
      ckpt_last_model.pth.tar
      ckpt_last_optimizer.pth.tar
      ckpt_best_acctop1_engine.pth.tar
      ckpt_best_acctop1_model.pth.tar
      ckpt_best_acctop1_optimizer.pth.tar
      logs.json
      logs.txt
      options.yaml
      view.html

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

Create a new :class:`torch.nn.Module` in `mnist/models/networks/my_net.py <https://github.com/Cadene/mnist.bootstrap.pytorch/tree/master/models/networks>`_.

.. code-block:: python

    import torch.nn as nn
    import torch.nn.functional as F

    class MyNet(nn.Module):

        def __init__(self, mul=2, drop=0.2):
            super(MyNet, self).__init__()
            self.mul = mul
            self.drop = drop
            self.conv1 = nn.Conv2d(1, 10*mul, kernel_size=5)
            self.conv2 = nn.Conv2d(10*mul, 20*mul, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320*mul, 50*mul)
            self.fc2 = nn.Linear(50*mul, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320*self.mul)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

Add a new options yaml file for it in `mnist/options/my_net.yaml <https://github.com/Cadene/mnist.bootstrap.pytorch/tree/master/options>`_:

.. code-block:: yaml
    :emphasize-lines: 16-18

    exp:
      dir: logs/mnist/my_net
      resume: # last, best_[...], or empty (from scratch)
    dataset:
      import: mnist.datasets.factory
      name: mnist
      dir: data/mnist
      train_split: train
      eval_split: val
      nb_threads: 4
      batch_size: 64
    model:
      name: simple
      network:
        import: mnist.models.networks.factory
        name: my_net
        mul: 2
        drop: 0.2
      criterion:
        name: nll
      metric:
        name: accuracy
        topk: [1,5]
    optimizer:
      name: sgd
      lr: 0.01
      momentum: 0.5
    engine:
      name: default
      debug: False
      nb_epochs: 10
      print_freq: 10
      saving_criteria:
      - loss:min          # save when new_best < best
      - accuracy_top1:max # save when new_best > best
      - accuracy_top5:max # save when new_best > best
    misc:
      cuda: False
      seed: 1337
    view:
    - logs:train_epoch.loss+logs:eval_epoch.loss
    - logs:train_batch.loss
    - logs:train_epoch.accuracy_top1+logs:eval_epoch.accuracy_top1
    - logs:train_epoch.accuracy_top5+logs:eval_epoch.accuracy_top5

We could also extend the current `mnist/options/abstract.yaml <https://github.com/Cadene/mnist.bootstrap.pytorch/tree/master/options/abstract.yaml>`_ options file:

.. code-block:: yaml

    __include__: abstract.yaml
    exp:
      dir: logs/mnist/my_net
    model:
      network:
        name: my_net
        mul: 2
        drop: 0.2

Finally, add your new network in the factory in `mnist/models/networks/factory.py <https://github.com/Cadene/mnist.bootstrap.pytorch/tree/master/models/networks>`_.

.. code-block:: python
  :emphasize-lines: 11-16

  from .net import Net
  from .my_net import MyNet

  def factory(engine=None):

    Logger()('Creating mnist network...')

    if Options()['model']['network']['name'] == 'net':
        network = Net()

    elif Options()['model']['network']['name'] == 'my_net':
        opt = Options()['model.network']
        network = MyNet(
            mul=opt['mul'],
            drop=opt['drop']
        )

    else:
        raise ValueError()

    if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    return network


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


