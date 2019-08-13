from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from bootstrap.engines.engine import Engine
from bootstrap.models.model import Model


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
