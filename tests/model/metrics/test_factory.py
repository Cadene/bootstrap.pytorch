import pytest

from bootstrap.lib.options import Options
from bootstrap.models.metrics.accuracy import Accuracy
from bootstrap.models.metrics.factory import factory


def test_initialize_options():
    options = {
        'model': {
            'metric': {},
        },
    }
    Options(options, run_parser=False)


def test_factory_incorrect_name():
    Options()['model']['metric'] = {'name': 'name'}
    with pytest.raises(ValueError):
        factory()


def test_factory_incorrect_option_1():
    Options()['model'] = {}
    criterion = factory()
    assert criterion is None


def test_factory_incorrect_option_2():
    Options()['model']['metric'] = None
    criterion = factory()
    assert criterion is None


def test_factory_accuracy_1():
    Options()['model']['metric'] = {'name': 'accuracy'}
    metric = factory()
    assert isinstance(metric, Accuracy)


def test_factory_accuracy_2():
    Options()['model']['metric'] = {
        'name': 'accuracy',
        'target_field_name': 'target',
        'input_field_name': 'input',
        'output_field_name': 'accuracy',
        'useless_option': 123,
    }
    metric = factory()
    assert isinstance(metric, Accuracy)
    assert metric.input_field_name == 'input'
    assert metric.output_field_name == 'accuracy'
    assert metric.target_field_name == 'target'


def test_reinitialize_options():
    options = {}
    Options(options, run_parser=False)
