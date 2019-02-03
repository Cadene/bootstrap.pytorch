<a href="http://remicadene.com/bootstrap"><img src="https://github.com/Cadene/bootstrap.pytorch/blob/master/docs/source/_static/img/bootstrap-logo-dark.png" width="50%"/></a>

<a href="https://travis-ci.org/Cadene/bootstrap.pytorch"><img src="https://api.travis-ci.org/Cadene/bootstrap.pytorch.svg?branch=master"/></a>

`Bootstrap` is a high-level framework for starting deep learning projects.
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

## Quick tour

To run an experiment (training + evaluation):
```
python -m bootstrap.run
       -o myproject/options/sgd.yaml
```

To display parsed options from the yaml file:
```
python -m bootstrap.run
       -o myproject/options/sgd.yaml
       -h
```

Running an experiment will create 4 files, here is an example with [mnist](https://github.com/Cadene/mnist.bootstrap.pytorch):

- [options.yaml](https://github.com/Cadene/bootstrap.pytorch/blob/master/docs/assets/logs/mnist/sgd/options.yaml) contains the options used for the experiment,
- [logs.txt](https://github.com/Cadene/bootstrap.pytorch/blob/master/docs/assets/logs/mnist/sgd/logs.txt) contains all the information given to the logger.
- [logs.json](https://github.com/Cadene/bootstrap.pytorch/blob/master/docs/assets/logs/mnist/sgd/logs.json) contains the following data: train_epoch.loss, train_batch.loss, eval_epoch.accuracy_top1, etc.
- <a href="http://htmlpreview.github.io/?https://raw.githubusercontent.com/Cadene/bootstrap.pytorch/master/docs/assets/logs/mnist/sgd/view.html">view.html</a> contains training and evaluation curves with javascript utilities (plotly).


To save the next experiment in a specific directory:
```
python -m bootstrap.run
       -o myproject/options/sgd.yaml
       --exp.dir logs/custom
```

To reload an experiment:
```
python -m bootstrap.run
       -o logs/custom/options.yaml
       --exp.resume last
```


## Documentation

The package reference is available on the [documentation website](http://remicadene.com/bootstrap).

It also contains some notes:

- [Installation](http://remicadene.com/bootstrap/#installation)
- [Concepts](http://remicadene.com/bootstrap/concepts.html)
- [Quickstart](http://remicadene.com/bootstrap/quickstart.html)
- [Directories](http://remicadene.com/bootstrap/directories.html)
- [Examples](http://remicadene.com/bootstrap/examples.html)

## Official project modules

- [mnist.bootstrap.pytorch](https://github.com/Cadene/mnist.bootstrap.pytorch) is a useful example for starting a quick project with bootstrap
- [vision.bootstrap.pytorch](https://github.com/Cadene/vision.bootstrap.pytorch) contains utilities to train image classifier, object detector, etc. on usual datasets like imagenet, cifar10, cifar100, coco, visual genome, etc.
- [recipe1m.bootstrap.pytorch](https://github.com/Cadene/recipe1m.bootstrap.pytorch) is a project for image-text retrieval related to the Recip1M dataset developped in the context of a [SIGIR18 paper](https://arxiv.org/abs/1804.11146).
- [block.bootstrap.pytorch](https://github.com/Cadene/block.bootstrap.pytorch) is a project focused on fusion modules related to the VQA 2.0, TDIUC and VRD datasets developped in the context of a [AAAI19 paper](http://remicadene.com/pdfs/paper_aaai2019.pdf).

## Poster

<a href="http://remicadene.com/bootstrap/_static/img/bootstrap_poster.pdf"><img src="http://remicadene.com/bootstrap/_static/img/bootstrap_poster_mini.png" width="20%"/></a>
