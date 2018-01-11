# bootstrap.pytorch

Bootstrap.pytorch is a highlevel framework for starting deep learning projects.
It aims at accelerate research projects and prototyping by providing a powerfull workflow which is easy to extend.
Bootstrap add almost zero layer of abstraction to pytorch.

*Few words from the authors (Remi Cadene, Micael Carvalho, Hedi Ben Younes, Thomas Robert): Bootstrap is the result of the time we spent engineering stuff since the beginning of our PhDs on different libraries and languages (Torch7, Keras, Tensorflow, Pytorch, Torchnet). It is also inspired by the modularity of modern web frameworks (TwitterBootstrap, CakePHP). We think that we were able to come up with a nice workflow and would like to open source it to get critics and to improve it furthermore with your help. Thanks!*

Coming soon:

- imagenet module
- finetuning module
- cross-modal retrieval module (Triplet loss)
- vqa module
- detection module (SSD, FasterRCNN)

## The principle

In a ideal world, one would only have to build a model (including criterion and metric) and a dataset to be able to run experiments.

Actually, bootstrap handles the rest! It contains:
- all the boilerplate code to run experiments,
- a clean way to organise your project,
- an options manager (parsing yaml file to generate command line arguments),
- a logger,
- saving/loading utilities,
- automatic curves ploting utilities with javascript support.


## Quick tour

To display options:
```
python main.py --path_opts mnist/options/sgd.yaml -h
> usage: main.py [-h] [--path_opts PATH_OPTS] [--misc.cuda [MISC.CUDA]]
> [...]
> optional arguments:
>  -h, --help            show this help message and exit
>  --path_opts PATH_OPTS
>                        path to a yaml file containing the default options
>                        (default: options/default.yaml)
>  --misc.cuda [MISC.CUDA]
>                        Default: False
> [...]
```

To run an experiment (training + evaluation):
```
python main.py --path_opts mnist/options/sgd.yaml
> [I 2018-01-11 13:22:37] ...rap/engines/factory.py.8: Creating engine...
> [I 2018-01-11 13:22:37] ...ap/datasets/factory.py.7: Creating dataset...
> [I 2018-01-11 13:22:37] ...ap/datasets/factory.py.20: train on trainset (60000 items)
> [I 2018-01-11 13:22:37] ...ap/datasets/factory.py.25: evaluate on valset (10000 items)
> [I 2018-01-11 13:22:37] ...trap/models/factory.py.10: Creating model...
> [I 2018-01-11 13:22:37] .../optimizers/factory.py.8: Creating optimizer...
> [I 2018-01-11 13:22:37] ...trap/engines/engine.py.60: Launching training procedures
> [I 2018-01-11 13:22:37] ...trap/engines/engine.py.82: Training model on trainset for epoch 0
> [I 2018-01-11 13:22:37] ...trap/engines/engine.py.138: train: epoch 0 | batch 0/937
> [I 2018-01-11 13:22:37] ...trap/engines/engine.py.141:       elapsed: 0:00:00 | left: 0:02:10
> [I 2018-01-11 13:22:37] ...trap/engines/engine.py.142:       process: 0.13902 | load: 0.07177
> [I 2018-01-11 13:22:37] ...trap/engines/engine.py.143:       loss: 2.37508
> [...]
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.209: eval: epoch 0 | batch 150/156
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.212:       elapsed: 0:00:00 | left: 0:00:00
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.213:       process: 0.00507 | load: 0.00101
> [S 2018-01-11 13:23:50] ...trap/engines/engine.py.66: eval_epoch.epoch: 0.00
> [S 2018-01-11 13:23:50] ...trap/engines/engine.py.66: eval_epoch.loss: 0.20
> [S 2018-01-11 13:23:50] ...trap/engines/engine.py.66: eval_epoch.accuracy_top5: 99.93
> [S 2018-01-11 13:23:50] ...trap/engines/engine.py.66: eval_epoch.accuracy_top1: 94.29
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.73: Saving last checkpoint
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.279: Saving model...
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.283: Saving optimizer...
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.287: Saving engine...
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.82: Training model on trainset for epoch 1
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.138: train: epoch 1 | batch 0/937
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.141:       elapsed: 0:00:00 | left: 0:01:19
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.142:       process: 0.08468 | load: 0.03953
> [I 2018-01-11 13:23:50] ...trap/engines/engine.py.143:       loss: 0.70270
> [...]
```

To save the next experiment in a specific directory:
```
python main.py --path_opts mnist/options/sgd.yaml --exp.dir logs/mnist/custom
```

To run with cuda:
```
python main.py --path_opts mnist/options/sgd.yaml \
--exp.dir logs/mnist/cuda --misc.cuda True
```

To create and visualize the training curves:
```
python view.py --path_opts logs/mnist/cuda/options.yaml
open logs/mnist/cuda/view.html
```

To visualize the log file because the server has been switched off:
```
less logs/mnist/cuda/logs.txt
> [...]
> [I 2018-01-11 13:25:55] ...trap/engines/engine.py.209: eval: epoch 9 | batch 100/156
> [I 2018-01-11 13:25:55] ...trap/engines/engine.py.212:       elapsed: 0:00:00 | left: 0:00:00
> [I 2018-01-11 13:25:55] ...trap/engines/engine.py.213:       process: 0.00244 | load: 0.00016
> [I 2018-01-11 13:25:55] ...trap/engines/engine.py.209: eval: epoch 9 | batch 110/156
> [I 2018-01-11 13:25:55] ...trap/engines/engine.py.212:       elapsed: 0:00:00 | left: 0:00:00
> [I 2018-01-11 13:25:55] ...trap/engines/engine.py.213:       process: 0.00248 | load: 0.00013
> [...]
```

To reload an experiment:
```
python main.py --path_opts logs/mnist/cuda/options.yaml --exp.resume last
```

To run an experiment on the training set only with an other set of options:
```
CUDA_VISIBLE_DEVICES=0 python main.py --path_opts mnist/options/adam.yaml \
--dataset.train_split train --dataset.eval_split
```

To evaluate the best model during the training (useful when the evaluation time is too high):
```
CUDA_VISIBLE_DEVICES=1 python main.py --path_opts logs/mnist/adam/options.yaml \
--exp.resume best_accuracy_top1 \
--dataset.train_split --dataset.eval_split val
```

To magically visualize the training and evaluation curves on the same view:
```
python view.py --path_opts logs/mnist/adam/options.yaml
open logs/mnist/adam/view.html
```

## Documentation

### Install

First install python 3 (we don't provide support for python 2). We advise you to install python 3 and pytorch with Anaconda:

- [python with anaconda](https://www.continuum.io/downloads)
- [pytorch with CUDA](http://pytorch.org/)

Bootstrap is not a lib such as pytorch, it is framework and thus need to be forked/cloned.

```
cd $HOME
git clone https://github.com/Cadene/bootstrap.pytorch.git 
cd bootstrap.pytorch
pip install -r requirements.txt
```

Then you can download any module you want. Let's install mnist:
```
git submodule update --init mnist
```


### Core bootstrap architecture

```
.
├── data                   # contains data only (raw and preprocessed)
├── logs                   # experiments dir (one dir per exp)
├── bootstrap      
|   ├── engines
|   |   ├── engine.py
|   |   └── factory.py
|   ├── datasets           # datasets classes & functions
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
|   |   └── factory.py
|   └── lib        
|       └── logger.py      # logs manager
|       └── options.py     # options manager
├── main.py                # train & eval models
└── view.py                # visualize logs and training curves
```

/!\ usually `data` contains symbolic links to directories stored on your fastest drive.


### Adding a project

Simply create a new module containing:

- your dataset in `myproject/datasets`,
- your model (network, criterion, metric) in `myproject/models`,
- your optimizer (if needed) in `myproject/optimizers`,
- your options in `options/myproject`.

We advise you to keep the same organization than in `bootstrap` directory and avoid modifying the core bootstrap files (`bootstrap/*.py`, `main.py`, `view.py`. Nevertheless, you are free to explore different ways.

```
.
├── data
├── logs
├── bootstrap              # don't modify, besides `factory.py` files
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
|   ├── engine             # if custom engine is needed
|   ├── optimizers         # if custom optimizer is needed
├── main.py                # avoid modifications to main
└── view.py                # avoid modifications to view
```

Some examples are available in the repository:
- mnist
- imagenet

### Tricks

Creating an experiment directory with the current datetime:
```
python main.py --path_opts mnist/options/sgd.yaml \
--exp.dir logs/mnist/`date "+%Y-%m-%d-%H-%M-%S"`_sgd
```


