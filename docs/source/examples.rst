Examples
========

We advise you to use tmux or screen to run experiments.

Running multiple experiments
----------------------------

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

.. code-block:: bash

    python -m bootstrap.compare
           -d logs/mnist/adam
              logs/mnist/sgd


Plotting logs manually
----------------------

.. code-block:: bash

    python -m bootstrap.views.view
           -o logs/mnist/sgd/options.yaml
    open logs/mnist/adam/view.html


Other tricks
------------

Creating an experiment directory with the current datetime:

.. code-block:: bash

    python -m bootstrap.run
           -o mnist/options/sgd.yaml
           --exp.dir logs/mnist/`date "+%Y-%m-%d-%H-%M-%S"`_sgd



