import pytest
import torch

from bootstrap.models.metrics.accuracy import Accuracy

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


def test_accuracy_target_field_topk_default():
    metric = Accuracy()
    assert metric.topk == [1, 5]


@pytest.mark.parametrize('topk', [[1], [1, 2, 3, 4, 5]])
def test_accuracy_target_field_topk(topk):
    metric = Accuracy(topk=topk)
    assert metric.topk == topk


def test_accuracy_target_field_name_default():
    metric = Accuracy()
    assert metric.target_field_name == 'class_id'


@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
def test_accuracy_target_field_name(target_field_name):
    metric = Accuracy(target_field_name=target_field_name)
    assert metric.target_field_name == target_field_name


def test_accuracy_input_field_name_default():
    metric = Accuracy()
    assert metric.input_field_name == 'net_out'


@pytest.mark.parametrize('input_field_name', ['net_out', 'input_metric'])
def test_accuracy_input_field_name(input_field_name):
    metric = Accuracy(input_field_name=input_field_name)
    assert metric.input_field_name == input_field_name


def test_accuracy_output_field_name_default():
    metric = Accuracy()
    assert metric.output_field_name == 'accuracy_top'


@pytest.mark.parametrize('output_field_name', ['accuracy_top', 'output_metric'])
def test_accuracy_output_field_name(output_field_name):
    metric = Accuracy(output_field_name=output_field_name)
    assert metric.output_field_name == output_field_name


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('output_field_name', ['accuracy_top', 'output_metric'])
def test_accuracy_forward_tensor_1(device, target_field_name, output_field_name):
    device = torch.device(device)
    metric = Accuracy(target_field_name=target_field_name,
                      output_field_name=output_field_name)
    net_out = torch.eye(3, 10).to(device=device)
    batch = {target_field_name: torch.arange(3).to(device=device)}
    cri_out = {}
    out = metric(cri_out, net_out, batch)
    assert out[f'{output_field_name}1'].item() == 100
    assert out[f'{output_field_name}5'].item() == 100


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('output_field_name', ['accuracy_top', 'output_metric'])
def test_accuracy_forward_tensor_2(device, target_field_name, output_field_name):
    device = torch.device(device)
    metric = Accuracy(target_field_name=target_field_name,
                      output_field_name=output_field_name)
    net_out = -torch.eye(3, 10).to(device=device)
    batch = {target_field_name: torch.arange(3).to(device=device)}
    cri_out = {}
    out = metric(cri_out, net_out, batch)
    assert out[f'{output_field_name}1'].item() == 0
    assert out[f'{output_field_name}5'].item() == 0


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
@pytest.mark.parametrize('output_field_name', ['accuracy_top', 'output_metric'])
def test_accuracy_forward_dict(device, target_field_name, input_field_name, output_field_name):
    device = torch.device(device)
    metric = Accuracy(target_field_name=target_field_name,
                      input_field_name=input_field_name,
                      output_field_name=output_field_name)
    net_out = {input_field_name: torch.eye(3, 10).to(device=device)}
    batch = {target_field_name: torch.arange(3).to(device=device)}
    cri_out = {}
    out = metric(cri_out, net_out, batch)
    assert out[f'{output_field_name}1'].item() == 100
    assert out[f'{output_field_name}5'].item() == 100


def test_accuracy_extra_args():
    Accuracy(reduction='sum', name='name', abc=123)
