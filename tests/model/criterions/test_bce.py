import math

import pytest
import torch

from bootstrap.models.criterions.bce import BCEWithLogitsLoss

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
def test_bce_target_field_name(target_field_name):
    loss = BCEWithLogitsLoss(target_field_name=target_field_name)
    assert loss.target_field_name == target_field_name


@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
def test_bce_input_field_name(input_field_name):
    loss = BCEWithLogitsLoss(input_field_name=input_field_name)
    assert loss.input_field_name == input_field_name


@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_bce_output_field_name(output_field_name):
    loss = BCEWithLogitsLoss(output_field_name=output_field_name)
    assert loss.output_field_name == output_field_name


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_bce_forward_tensor(device, target_field_name, output_field_name):
    device = torch.device(device)
    loss = BCEWithLogitsLoss(target_field_name=target_field_name,
                             output_field_name=output_field_name).to(device=device)
    net_out = torch.ones(3, 4).to(device=device)
    batch = {target_field_name: torch.eye(3, 4).to(device=device)}
    out = loss(net_out, batch)
    s1 = math.exp(1) / (1 + math.exp(1))
    s2 = 1 / (1 + math.exp(1))
    gt = - (3 * math.log(s1) + 9 * math.log(s2)) / 12
    assert math.isclose(out[output_field_name].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_bce_forward_dict(device, target_field_name, input_field_name, output_field_name):
    device = torch.device(device)
    loss = BCEWithLogitsLoss(target_field_name=target_field_name,
                             input_field_name=input_field_name,
                             output_field_name=output_field_name).to(device=device)
    net_out = {input_field_name: torch.ones(3, 4).to(device=device)}
    batch = {target_field_name: torch.eye(3, 4).to(device=device)}
    out = loss(net_out, batch)
    s1 = math.exp(1) / (1 + math.exp(1))
    s2 = 1 / (1 + math.exp(1))
    gt = - (3 * math.log(s1) + 9 * math.log(s2)) / 12
    assert math.isclose(out[output_field_name].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('dim', [1, 2, 4])
def test_bce_forward_batch_size(device, batch_size, dim):
    device = torch.device(device)
    loss = BCEWithLogitsLoss().to(device=device)
    net_out = torch.ones(batch_size, dim).to(device=device)
    batch = {'class_id': torch.eye(batch_size, dim).to(device=device)}
    out = loss(net_out, batch)
    s1 = math.exp(1) / (1 + math.exp(1))
    s2 = 1 / (1 + math.exp(1))
    n = min(batch_size, dim)
    gt = - (n * math.log(s1) + (batch_size * dim - n) * math.log(s2)) / (batch_size * dim)
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_bce_forward_reduction_mean(device):
    device = torch.device(device)
    loss = BCEWithLogitsLoss(reduction='mean').to(device=device)
    net_out = torch.ones(3, 4).to(device=device)
    batch = {'class_id': torch.eye(3, 4).to(device=device)}
    out = loss(net_out, batch)
    s1 = math.exp(1) / (1 + math.exp(1))
    s2 = 1 / (1 + math.exp(1))
    gt = - (3 * math.log(s1) + 9 * math.log(s2)) / 12
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_bce_forward_reduction_sum(device):
    device = torch.device(device)
    loss = BCEWithLogitsLoss(reduction='sum').to(device=device)
    net_out = torch.ones(3, 4).to(device=device)
    batch = {'class_id': torch.eye(3, 4).to(device=device)}
    out = loss(net_out, batch)
    s1 = math.exp(1) / (1 + math.exp(1))
    s2 = 1 / (1 + math.exp(1))
    gt = - (3 * math.log(s1) + 9 * math.log(s2))
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


def test_bce_extra_args():
    BCEWithLogitsLoss(reduction='sum', name='name', abc=123)
