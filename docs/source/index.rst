Bootstrap Documentation
=======================

:mod:`Bootstrap` is a high-level framework for starting deep learning projects.
It aims at accelerating research projects and prototyping by providing a powerful workflow focused on your dataset and model only.

And it is:

- Scalable
- Modular
- Shareable
- Extendable
- Uncomplicated
- Built for reproducibility
- Easy to log and plot anything

It's not a wrapper over pytorch, it's a powerful extension.

Installation
============

First, install python3 and pytorch with Anaconda:

- `python with anaconda <https://www.continuum.io/downloads>`_
- `pytorch with CUDA <http://pytorch.org>`_

We advise you to clone bootstrap to start a new project. By doing so, it is easier to prototype and debug your code thanks to a direct access to bootstrap core functions:

.. code:: bash

    git clone https://github.com/Cadene/bootstrap.pytorch.git
    cd bootstrap.pytorch
    pip install -r requirements.txt

Using bootstrap like a python library is also possible. You can use pip install:

.. code:: bash

    pip install bootstrap.pytorch

Or install from source:

.. code:: bash

    git clone https://github.com/Cadene/bootstrap.pytorch.git
    cd bootstrap.pytorch
    python setup.py install


.. toctree::
   :maxdepth: 2
   :caption: Notes

   concepts
   quickstart
   directories
   examples 

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   engines
   models
   networks
   criterions
   metrics
   datasets
   optimizers
   views
   options
   logger
   lib

.. toctree::
   :maxdepth: 2
   :caption: Package Reference for Submodules
    
   submodules/mnist/models/networks

.. automodule:: bootstrap
   :members:

Few words from the authors
--------------------------

Bootstrap is the result of the time we spent engineering stuff since the beginning of our PhDs on different libraries and languages (Torch7, Keras, Tensorflow, Pytorch, Torchnet). It is also inspired by the modularity of modern web frameworks. We think that we were able to come up with a nice workflow and good practicies that we would like to share. Criticisms are welcome. Thanks for considering our work!
