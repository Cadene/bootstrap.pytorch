Running train and eval asynchronously
-------------------------------------

To run an experiment on the training set only with an other set of options:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=0
    python -m bootstrap.run
           -o mnist/options/sgd.yaml
           --exp.dir logs/mnist/sgd
           --dataset.train_split train
           --dataset.eval_split

Then, we must trigger the asynchronous evaluation process of the last checkpoint at the end of a training epoch.

TODO:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=1
    python -m bootstrap.run
           -o logs/mnist/sgd/options.yaml
           --exp.resume last
           --dataset.train_split
           --dataset.eval_split val