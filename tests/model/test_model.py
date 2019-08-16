from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from bootstrap.engines.engine import Engine
from bootstrap.models.metrics.accuracy import Accuracy
from bootstrap.models.model import Model


def test_init_network_linear():
    """ Test network initialization with linear
    """
    network = nn.Linear(in_features=2, out_features=4)
    model = Model(network=network)
    assert isinstance(model.network, nn.Linear)


def test_init_network_conv2d():
    """ Test network initialization with conv2d
    """
    network = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
    model = Model(network=network)
    assert isinstance(model.network, nn.Conv2d)


def test_init_criterions_bce_nll():
    """ Test criterions initialization: BCE anf NLL
    """
    criterions = {'train': nn.BCELoss(), 'eval': nn.NLLLoss()}
    model = Model(criterions=criterions)
    assert isinstance(model.criterions['train'], nn.BCELoss)
    assert isinstance(model.criterions['eval'], nn.NLLLoss)


def test_init_criterions_nll_bce():
    """ Test criterions initialization: NLL and BCE
    """
    criterions = {'train': nn.NLLLoss(), 'eval': nn.BCELoss()}
    model = Model(criterions=criterions)
    assert isinstance(model.criterions['train'], nn.NLLLoss)
    assert isinstance(model.criterions['eval'], nn.BCELoss)


def test_init_metrics_nll_accuracy():
    """ Test metrics initialization: accuracy and NLL
    """
    metrics = {'train': Accuracy(), 'eval': nn.NLLLoss()}
    model = Model(metrics=metrics)
    assert isinstance(model.metrics['train'], Accuracy)
    assert isinstance(model.metrics['eval'], nn.NLLLoss)


def test_init_metrics_accuracy_nll():
    """ Test metrics initialization: NLL and accuracy
    """
    metrics = {'train': nn.NLLLoss(), 'eval': Accuracy()}
    model = Model(metrics=metrics)
    assert isinstance(model.metrics['train'], nn.NLLLoss)
    assert isinstance(model.metrics['eval'], Accuracy)


def test_eval():
    """ Test eval method of a model
    """
    model = Model()
    model.eval()
    assert model.mode == 'eval'


def test_eval_with_network():
    """ Test eval method of a model with a network
    """
    network = nn.Linear(2, 4)
    model = Model(network=network)
    model.eval()
    assert not model.network.training


def test_train():
    """ Test train method of a model
    """
    model = Model()
    model.train()
    assert model.mode == 'train'


def test_train_with_network():
    """ Test train method of a model with a network
    """
    network = nn.Linear(2, 4)
    model = Model(network=network)
    model.train()
    assert model.network.training


if torch.cuda.is_available():
    def test_cuda():
        """ Test cuda method of a model
        """
        model = Model()
        model.cuda()
        assert model.is_cuda


def test_cpu():
    """ Test cpu method of a model
    """
    model = Model()
    model.cpu()
    assert not model.is_cuda


def test_forward_no_network():
    x = torch.randn(4, 2)
    model = Model()
    with pytest.raises(TypeError):  # no network defined
        model(x)


# def test_forward_network():
#     x = torch.randn(4, 2)
#     network = nn.Linear(2, 4)
#     model = Model(network=network)
#     y = model(x)
#     print(y)
#     assert y.size() == torch.Size([4, 4])


# def test_forward_network_criterion():
#     x = torch.randn(4, 2)
#     network = nn.Linear(2, 4)
#     criterions = {'train': nn.BCELoss(), 'eval': nn.BCELoss()}
#     model = Model(network=network, criterions=criterions)
#     batch = {'': x, }
#     y = model(x)
#     print(y)
#     assert y.size() == torch.Size([4, 4])


def test_state_dict_network():
    """ Test state_dict method for a model that contains a network without criterion and metric.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    model = Model(engine=engine, network=network)

    state = model.state_dict()

    assert any(state['network'])
    assert list(state['network'].items())[0][1].shape == torch.Size([4, 2])
    assert not any(state['criterions'])
    assert not any(state['metrics'])


def test_state_dict_criterion_without_parameter():
    """ Test state_dict method for a model that contains a criterion without learnable parameter.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    criterions = {'train': nn.BCELoss()}
    model = Model(engine=engine, network=network, criterions=criterions)

    state = model.state_dict()

    assert any(state['network'])
    assert list(state['network'].items())[0][1].shape == torch.Size([4, 2])
    assert not any(state['criterions']['train'])
    with pytest.raises(KeyError):
        state['criterions']['eval']
    assert not any(state['metrics'])


def test_state_dict_criterion_with_parameter():
    """ Test state_dict method for a model that contains a criterion with learnable parameters.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    # linear is not really a criterion, but it contains learnable parameters ;)
    criterions = {'train': nn.Linear(2, 3, bias=False), 'eval': nn.BCELoss()}
    model = Model(engine=engine, network=network, criterions=criterions)

    state = model.state_dict()

    assert any(state['network'])
    assert list(state['network'].items())[0][1].shape == torch.Size([4, 2])
    assert list(state['criterions']['train'].items())[0][1].shape == torch.Size([3, 2])
    assert not any(state['criterions']['eval'])
    assert not any(state['metrics'])


def test_state_dict_metric_without_parameter():
    """ Test state_dict method for a model that contains a metric without learnable parameter.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    # NLLLoss is not really a metric ;)
    metrics = {'train': nn.NLLLoss()}
    model = Model(engine=engine, network=network, metrics=metrics)

    state = model.state_dict()

    assert any(state['network'])
    assert list(state['network'].items())[0][1].shape == torch.Size([4, 2])
    assert not any(state['criterions'])
    assert not any(state['metrics']['train'])
    with pytest.raises(KeyError):
        state['metrics']['eval']


def test_state_dict_network_criterion_metric_with_parameter():
    """ Test state_dict method for a model that contains a metric with learnable parameters.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    # linear is not really a metric, but it contains learnable parameters ;)
    metrics = {'train': nn.Linear(3, 2, bias=False), 'eval': nn.NLLLoss()}
    model = Model(engine=engine, network=network, metrics=metrics)

    state = model.state_dict()

    assert any(state['network'])
    assert list(state['network'].items())[0][1].shape == torch.Size([4, 2])
    assert not any(state['criterions'])
    assert list(state['metrics']['train'].items())[0][1].shape == torch.Size([2, 3])
    assert not any(state['metrics']['eval'])


def test_load_state_dict_network():
    """ Test state_dict method for a model that contains a network without criterion and metric.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    model = Model(engine=engine, network=network)

    weight = torch.ones(4, 2)
    state = {'network': OrderedDict([('weight', weight)]), 'criterions': {}, 'metrics': {}}

    model.load_state_dict(state)

    assert torch.equal(model.network.weight, weight)


def test_load_state_dict_criterions():
    """ Test state_dict method for a model that contains a network without criterion and metric.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    criterions = {'train': nn.Linear(2, 3, bias=False), 'eval': nn.BCELoss()}
    model = Model(engine=engine, network=network, criterions=criterions)

    weight = torch.ones(3, 2)
    state = {'network': OrderedDict([('weight', torch.ones(4, 2))]),
             'criterions': {'train': OrderedDict([('weight', weight)]), 'eval': {}}}

    model.load_state_dict(state)

    assert torch.equal(model.criterions['train'].weight, weight)


def test_load_state_dict_metrics():
    """ Test state_dict method for a model that contains a network without criterion and metric.
    """
    engine = Engine()
    network = nn.Linear(2, 4, bias=False)
    metrics = {'train': nn.Linear(2, 3, bias=False), 'eval': nn.BCELoss()}
    model = Model(engine=engine, network=network, metrics=metrics)

    weight = torch.ones(3, 2)
    state = {'network': OrderedDict([('weight', torch.ones(4, 2))]),
             'metrics': {'train': OrderedDict([('weight', weight)]), 'eval': {}}}

    model.load_state_dict(state)

    assert torch.equal(model.metrics['train'].weight, weight)
