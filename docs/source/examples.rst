Examples
========

Existing modules
----------------

- Mnist Example in `mnist.bootstrap.pytorch <https://github.com/Cadene/mnist.bootstrap.pytorch>`_
- Image Classification in `imclassif.bootstrap.pytorch <https://github.com/Cadene/imclassif.bootstrap.pytorch>`_
- Image To Recipe Convertor in `recipe1m.bootstrap.pytorch <https://github.com/Cadene/recipe1m.bootstrap.pytorch>`_

To come:

- Visual Question Answering in `vqa.bootstrap.pytorch <https://github.com/Cadene/vqa.bootstrap.pytorch>`_
- Visual Relationship Detection in `rel.bootstrap.pytorch <https://github.com/Cadene/rel.bootstrap.pytorch>`_
- Neural Architecture Search in `nas.bootstrap.pytorch <https://github.com/Cadene/nas.bootstrap.pytorch>`_
- Metric Learning in `retrieval.bootstrap.pytorch <https://github.com/Cadene/retrieval.bootstrap.pytorch>`_
- Object Detection in `detection.bootstrap.pytorch <https://github.com/Cadene/detection.bootstrap.pytorch>`_

Running multiple experiments
----------------------------

We advise you to use `tmux <https://github.com/tmux/tmux/wiki>`_ or `screen <https://www.gnu.org/software/screen/manual/screen.html>`_ to run experiments.

.. code-block:: bash

    list_lr=(0.1 0.08 0.06 0.04)
    gpu=0
    for lr in $list_lr
    do
      CUDA_VISIBLE_DEVICES=$gpu python -m bootstrap.run
            -o mnist/options/sgd.yaml
            --optimizer.lr 
            --exp.dir logs/mnist/lr_$lr &
      gpu = $(( $gpu + 1 ))
    done


.. code-block:: bash

    tail logs/mnist/lr_0.1/logs.txt


Comparing experiments
---------------------

We provide some utility functions to compare multiple experiments based on their best score for specific metrics.

.. code-block:: bash

    python -m bootstrap.compare -h
    usage: compare.py [-h] [-n NB_EPOCHS] [-d [DIR_LOGS [DIR_LOGS ...]]]
                      [-k metric order]

    optional arguments:
      -h, --help            show this help message and exit
      -n NB_EPOCHS, --nb_epochs NB_EPOCHS
      -d [DIR_LOGS [DIR_LOGS ...]], --dir_logs [DIR_LOGS [DIR_LOGS ...]]
      -k metric order, --keys metric order

.. code-block:: bash

    python -m bootstrap.compare
           -d logs/mnist/adam
              logs/mnist/sgd
    > Metric: eval_epoch.accuracy_top1

        Place  Method      Score    Epoch
      -------  --------  -------  -------
            1  sgd       98.4773        9
            2  adam      98.3212        9


Plotting logs manually
----------------------

The plotting utilities of :mod:`Bootstrap` are included in `bootstrap/views <https://github.com/Cadene/bootstrap.pytorch/tree/master/bootstrap/views>`_. A :class:`bootstrap.views.view.View` is created during the initialization of the :class:`bootstrap.engines.engine.Engine`. It is used to generate automaticaly a `view.html <https://rawgit.com/Cadene/bootstrap.pytorch/master/logs/mnist/sgd/view.html>`_ file after each training or evaluation epoch. Nevertheless, you can call it manually by doing so:

.. code-block:: bash

    python -m bootstrap.views.view
           -o logs/mnist/sgd/options.yaml
    open logs/mnist/sgd/view.html


Other tricks
------------

Creating an experiment directory with the current datetime:

.. code-block:: bash

    python -m bootstrap.run
           -o mnist/options/sgd.yaml
           --exp.dir logs/mnist/`date "+%Y-%m-%d-%H-%M-%S"`_sgd



