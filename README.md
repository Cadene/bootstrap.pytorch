# bootstrap.pytorch

Bootstrap.pytorch is a highlevel framework for starting deep learning projects.
It aims at accelerating research projects and prototyping by providing a powerfull workflow which is as flexible as possible and easy to extend.
Bootstrap add almost zero layer of abstraction to pytorch.

*Few words from the authors (Remi Cadene, Micael Carvalho, Hedi Ben Younes, Thomas Robert): Bootstrap is the result of the time we spent engineering stuff since the beginning of our PhDs on different libraries and languages (Torch7, Keras, Tensorflow, Pytorch, Torchnet). It is also inspired by the modularity of modern web frameworks (TwitterBootstrap, CakePHP). We think that we were able to come up with a nice workflow and good practicies that we would like to share. Criticisms are welcome. Thanks for considering our work!*

**Coming soon**:

- better documentation
- imagenet module
- finetuning module
- cross-modal retrieval module (Triplet loss)
- vqa module
- detection module (SSD, FasterRCNN)
- docker support

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

To display parsed options from the yaml file:
```
python main.py --path_opts mnist/options/sgd.yaml -h
```

To run an experiment (training + evaluation):
```
python main.py --path_opts mnist/options/sgd.yaml
```

Running an experiment will create 3 files:

- [options.yaml](https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/mnist/sgd/options.yaml) contains the options used for the experiment,
- [logs.txt](https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/mnist/sgd/logs.txt) contains all the information given to the logger.
- [logs.json](https://github.com/Cadene/bootstrap.pytorch/blob/master/logs/mnist/sgd/logs.json) contains the following data: train_epoch.loss, train_batch.loss, eval_epoch.accuracy_top1, etc.


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

Running `view.py` over an experiment will create an html file containing training and evaluation curves. An example is available here: <a href="https://rawgit.com/Cadene/bootstrap.pytorch/master/logs/mnist/sgd/view.html">view.html</a>

To reload an experiment:
```
python main.py --path_opts logs/mnist/cuda/options.yaml --exp.resume last
```

## The good practices used to build Bootstrap

**General**

- modularity and code normalization (but not too much) to ensure code reusability
- CTRL+V are faster and clearer than bad refactoring

**Options**

- options are stored in a single place in a yaml file
- yaml options files can be extended as in OOP to avoid mistakes when prototyping
- options from yaml files are parsed to create command line arguments and thus can be overwritten by command line to ensure easy hyper parameters search
- loaded options are accessible everywhere via a global variable to ensure fast and easy prototyping

**Logs**

- everything related to an experiment is stored in the same directory (logs, checkpoints, visualizations, options, etc.)
- the experiment directores can be named as desired (including datetime, hyperparamters, server name, etc.) to ensure a personalized workflow
- the print function is replaced by a logger to be able to add useful information such as line number and file name where the log function has been called
- the log function write everythings in a text file to ensure that you can access to the full workflow trace at any time
- information such as the loss per epoch, metrics per epoch, loss per batch, metrics per batch are automatically saved in logs.json
- the logger which manages prints, logs.txt and logs.json is accessible everywhere via a global variable to ensure fast and easy debugging (example: you want to log the output Tensor of a certain layer of your network along the evaluation process)


**Checkpoints**

- experiments are saved after every epoch with multiple checkpoints (last and best)
- multiple criteria can be used to save checkpoints (ex: best accuracy top1, best loss, etc.)
- checkpoints are divided in several parts to ensure that you can delete the optimizer state only to save memory space
- experiments can be stopped and started again using only two commands: path to experiment and name of checkpoint to be resumed

**Engine**

- a per epoch engine able to train and evaluate the model after each epochs is used by default
- hooks functions trigered by the engine can be registered by the model (network, criterion, metric) and the optimizer (example: your mAP metric need to calculate the mAP at the end of the evaluation epoch, so it would register a hook to do it)
- easy to extend engine, model and optimizer to be able to build a fully personnalized workflow (example: if you need to save checkpoints during an epoch you can overwrite methods of the default engine)

**Dataset**

- dataset classes handle automatic download and preprocessing to be able to run an experiment with only one command after having installed bootstrap
- a batch returned by a dataset is a dictionnary (possibly nested) having the network input and the criterion target as well as useful information on items (such as the image path, the item index, etc.)
- this dictionnary (e.g. batch) is the only input of the model and is by default transferred to the network, criterion and metric ; this ensure easy debugging and easy visualization if needed

**Model**

- the model is made of a network (ex: resnet152), criterions (ex: nll), metrics (ex: accuracy top1/top5)
- the model is in charge of converting the input to torch.cuda.Tensor (if needed) and to torch.autograd.Variable ; this ensure that you only need to load the model to be able to feed it any input you want (easy debugging, visualization, and deployment)


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

We advise you to keep the same organization than in `bootstrap` directory and avoid modifying the core bootstrap files (`bootstrap/*.py`, `main.py`, `view.py`). Nevertheless, you are free to explore different ways.

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

To visualize the log file because the server has been switched off:
```
less logs/mnist/cuda/logs.txt
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
