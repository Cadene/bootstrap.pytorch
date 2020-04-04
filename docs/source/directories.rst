Directories
===========

Core bootstrap architecture
---------------------------

.. code:: bash

    .
    ├── bootstrap   
    |   ├── run.py             # train & eval models
    |   ├── compare.py         # compare experiments
    |   ├── engines
    |   |   ├── engine.py
    |   |   └── factory.py
    |   ├── datasets           # datasets classes & functions
    |   |   ├── dataset.py
    |   |   ├── transforms.py
    |   |   ├── utils.py
    |   |   └── factory.py
    |   ├── models
    |   |   ├── model.py
    |   |   ├── factory.py
    |   |   ├── networks
    |   |   |   └── factory.py
    |   |   ├── criterions
    |   |   |   ├── nll.py
    |   |   |   ├── cross_entropy.py
    |   |   |   └── factory.py
    |   |   └── metrics
    |   |       ├── accuracy.py
    |   |       └── factory.py
    |   ├── optimizers 
    |   |   ├── grad_clipper.py
    |   |   ├── lr_scheduler.py
    |   |   └── factory.py
    |   ├── options
    |   |   ├── abstract.yaml   # example of default options
    |   |   └── example.yaml
    |   ├── views             # plotting utilities
    |   |   ├── view.py
    |   |   └── factory.py
    |   └── lib        
    |       └── logger.py      # logs manager
    |       └── options.py     # options manager
    ├── data                   # contains data only (raw and preprocessed)
    └── logs                   # experiments dir (one directory per experiment)

/!\ usually `data` and `logs` contain symbolic links to directories stored on your fastest drive.


Project architecture
--------------------

Simply create a new module containing:

- your dataset in `myproject/datasets`,
- your model (network, criterion, metric) in `myproject/models`,
- your optimizer (if needed) in `myproject/optimizers`,
- your options in `myproject/options`.

We advise you to keep the same organization than in `bootstrap` directory and avoid modifying the core bootstrap files (`bootstrap/*.py`, `main.py`, `view.py`). Nevertheless, you are free to explore different ways.

.. code:: bash

    .
    ├── bootstrap              # avoid modifications to bootstrap/*
    ├── myproject
    |   ├── options            # add your yaml files here
    |   ├── datasets          
    |   |   ├── mydataset.py
    |   |   └── factory.py     # init dataset with your options
    |   ├── models 
    |   |   ├── model.py       # if custom model is needed
    |   |   ├── factory.py     # if custom model is needed
    |   |   ├── networks
    |   |   |   ├── mynetwork.py
    |   |   |   └── factory.py
    |   |   ├── criterions     # if custom criterion is needed
    |   |   |   ├── mycriterion.py
    |   |   |   └── factory.py
    |   |   └── metrics        # if custom metric is needed
    |   |       ├── mymetric.py
    |   |       └── factory.py
    |   ├── engines            # if custom engine is needed
    |   ├── optimizers         # if custom optimizer is needed
    |   └── views              # if custom plotting is needed
    ├── data
    └── logs

Some examples are available in the repository:
- mnist
