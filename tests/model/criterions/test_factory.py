import pytest

from bootstrap.lib.options import Options
from bootstrap.models.criterions.bce import BCEWithLogitsLoss
from bootstrap.models.criterions.cross_entropy import CrossEntropyLoss
from bootstrap.models.criterions.factory import factory
from bootstrap.models.criterions.nll import NLLLoss


def test_initialize_options():
    options = {
        'model': {
            'criterion': {},
        },
    }
    Options(options, run_parser=False)


def test_factory_incorrect_name():
    Options()['model']['criterion'] = {'name': 'name'}
    with pytest.raises(ValueError):
        factory()


def test_factory_incorrect_option_1():
    Options()['model'] = {}
    criterion = factory()
    assert criterion is None


def test_factory_incorrect_option_2():
    Options()['model']['criterion'] = None
    criterion = factory()
    assert criterion is None


def test_factory_nll_1():
    Options()['model']['criterion'] = {'name': 'nll'}
    criterion = factory()
    assert isinstance(criterion, NLLLoss)


def test_factory_nll_2():
    Options()['model']['criterion'] = {
        'name': 'nll',
        'target_field_name': 'target',
        'input_field_name': 'input',
        'output_field_name': 'nll_loss',
        'useless_option': 123,
    }
    criterion = factory()
    assert isinstance(criterion, NLLLoss)
    assert criterion.input_field_name == 'input'
    assert criterion.output_field_name == 'nll_loss'
    assert criterion.target_field_name == 'target'


def test_factory_bce_with_logits_1():
    Options()['model']['criterion'] = {'name': 'BCEWithLogitsLoss'}
    criterion = factory()
    assert isinstance(criterion, BCEWithLogitsLoss)


def test_factory_bce_with_logits_2():
    Options()['model']['criterion'] = {
        'name': 'BCEWithLogitsLoss',
        'target_field_name': 'target',
        'input_field_name': 'input',
        'output_field_name': 'bce_with_logits_loss',
        'useless_option': 123,
    }
    criterion = factory()
    assert isinstance(criterion, BCEWithLogitsLoss)
    assert criterion.input_field_name == 'input'
    assert criterion.output_field_name == 'bce_with_logits_loss'
    assert criterion.target_field_name == 'target'


def test_factory_cross_entropy_1():
    Options()['model']['criterion'] = {'name': 'cross_entropy'}
    criterion = factory()
    assert isinstance(criterion, CrossEntropyLoss)


def test_factory_cross_entropy_2():
    Options()['model']['criterion'] = {
        'name': 'cross_entropy',
        'target_field_name': 'target',
        'input_field_name': 'input',
        'output_field_name': 'cross_entropy_loss',
        'useless_option': 123,
    }
    criterion = factory()
    assert isinstance(criterion, CrossEntropyLoss)
    assert criterion.input_field_name == 'input'
    assert criterion.output_field_name == 'cross_entropy_loss'
    assert criterion.target_field_name == 'target'


def test_reinitialize_options():
    options = {}
    Options(options, run_parser=False)
