import math

import pytest
import torch

from bootstrap.models.criterions.cross_entropy import CrossEntropyLoss

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


def test_cross_entropy_target_field_name_default():
    loss = CrossEntropyLoss()
    assert loss.target_field_name == 'class_id'


@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
def test_cross_entropy_target_field_name(target_field_name):
    loss = CrossEntropyLoss(target_field_name=target_field_name)
    assert loss.target_field_name == target_field_name


def test_cross_entropy_input_field_name_default():
    loss = CrossEntropyLoss()
    assert loss.input_field_name == 'net_out'


@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
def test_cross_entropy_input_field_name(input_field_name):
    loss = CrossEntropyLoss(input_field_name=input_field_name)
    assert loss.input_field_name == input_field_name


def test_cross_entropy_output_field_name_default():
    loss = CrossEntropyLoss()
    assert loss.output_field_name == 'loss'


@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_cross_entropy_output_field_name(output_field_name):
    loss = CrossEntropyLoss(output_field_name=output_field_name)
    assert loss.output_field_name == output_field_name


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_cross_entropy_forward_tensor(device, target_field_name, output_field_name):
    device = torch.device(device)
    loss = CrossEntropyLoss(target_field_name=target_field_name,
                            output_field_name=output_field_name).to(device=device)
    net_out = torch.zeros(3, 4).to(device=device)
    batch = {target_field_name: torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = -  math.log(1 / 4)
    assert math.isclose(out[output_field_name].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_cross_entropy_forward_dict(device, target_field_name, input_field_name, output_field_name):
    device = torch.device(device)
    loss = CrossEntropyLoss(target_field_name=target_field_name,
                            input_field_name=input_field_name,
                            output_field_name=output_field_name).to(device=device)
    net_out = {input_field_name: torch.ones(3, 4).to(device=device)}
    batch = {target_field_name: torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = -  math.log(1 / 4)
    assert math.isclose(out[output_field_name].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('dim', [1, 2, 4])
def test_cross_entropy_forward_batch_size(device, batch_size, dim):
    device = torch.device(device)
    loss = CrossEntropyLoss().to(device=device)
    net_out = torch.zeros(batch_size, dim).to(device=device)
    batch = {'class_id': torch.zeros(batch_size).long().to(device=device)}
    out = loss(net_out, batch)
    gt = -  math.log(1 / dim)
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_cross_entropy_forward_reduction_mean(device):
    device = torch.device(device)
    loss = CrossEntropyLoss(reduction='mean').to(device=device)
    net_out = torch.zeros(3, 4).to(device=device)
    batch = {'class_id': torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = -  math.log(1 / 4)
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_cross_entropy_forward_reduction_sum(device):
    device = torch.device(device)
    loss = CrossEntropyLoss(reduction='sum').to(device=device)
    net_out = torch.zeros(3, 4).to(device=device)
    batch = {'class_id': torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = - 3 * math.log(1 / 4)
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


def test_cross_entropy_extra_args():
    CrossEntropyLoss(reduction='sum', name='name', abc=123)
