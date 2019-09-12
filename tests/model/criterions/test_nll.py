import math

import pytest
import torch

from bootstrap.models.criterions.nll import NLLLoss

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


def test_nll_target_field_name_default():
    loss = NLLLoss()
    assert loss.target_field_name == 'class_id'


@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
def test_nll_target_field_name(target_field_name):
    loss = NLLLoss(target_field_name=target_field_name)
    assert loss.target_field_name == target_field_name


def test_nll_input_field_name_default():
    loss = NLLLoss()
    assert loss.input_field_name == 'net_out'


@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
def test_nll_input_field_name(input_field_name):
    loss = NLLLoss(input_field_name=input_field_name)
    assert loss.input_field_name == input_field_name


def test_nll_output_field_name_default():
    loss = NLLLoss()
    assert loss.output_field_name == 'loss'


@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_nll_output_field_name(output_field_name):
    loss = NLLLoss(output_field_name=output_field_name)
    assert loss.output_field_name == output_field_name


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_nll_forward_tensor(device, target_field_name, output_field_name):
    device = torch.device(device)
    loss = NLLLoss(target_field_name=target_field_name,
                   output_field_name=output_field_name).to(device=device)
    net_out = - 0.5 * torch.ones(3, 4).to(device=device)
    batch = {target_field_name: torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = 0.5
    assert math.isclose(out[output_field_name].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('target_field_name', ['class_id', 'target'])
@pytest.mark.parametrize('input_field_name', ['net_out', 'input_loss'])
@pytest.mark.parametrize('output_field_name', ['loss', 'output_loss'])
def test_nll_forward_dict(device, target_field_name, input_field_name, output_field_name):
    device = torch.device(device)
    loss = NLLLoss(target_field_name=target_field_name,
                   input_field_name=input_field_name,
                   output_field_name=output_field_name).to(device=device)
    net_out = {input_field_name: - 0.5 * torch.ones(3, 4).to(device=device)}
    batch = {target_field_name: torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = 0.5
    assert math.isclose(out[output_field_name].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('dim', [1, 2, 4])
def test_nll_forward_batch_size(device, batch_size, dim):
    device = torch.device(device)
    loss = NLLLoss().to(device=device)
    net_out = - torch.ones(batch_size, dim).to(device=device) / (dim + 1)
    batch = {'class_id': torch.zeros(batch_size).long().to(device=device)}
    out = loss(net_out, batch)
    gt = 1 / (dim + 1)
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_nll_forward_reduction_mean(device):
    device = torch.device(device)
    loss = NLLLoss(reduction='mean').to(device=device)
    net_out = - 0.5 * torch.ones(3, 4).to(device=device)
    batch = {'class_id': torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = 0.5
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_nll_forward_reduction_sum(device):
    device = torch.device(device)
    loss = NLLLoss(reduction='sum').to(device=device)
    net_out = - 0.5 * torch.ones(3, 4).to(device=device)
    batch = {'class_id': torch.tensor([0, 1, 2]).to(device=device)}
    out = loss(net_out, batch)
    gt = 3 * 0.5
    assert math.isclose(out['loss'].cpu().item(), gt, abs_tol=1e-6)


@pytest.mark.parametrize('device', devices)
def test_nll_forward_incorrect_input_1(device):
    device = torch.device(device)
    loss = NLLLoss().to(device=device)
    net_out = {'input': - 0.5 * torch.ones(3, 4).to(device=device)}
    batch = {'class_id': torch.tensor([0, 1, 2]).to(device=device)}
    with pytest.raises(KeyError):
        loss(net_out, batch)


@pytest.mark.parametrize('device', devices)
def test_nll_forward_incorrect_input_2(device):
    device = torch.device(device)
    loss = NLLLoss().to(device=device)
    net_out = [- 0.5 * torch.ones(3, 4).to(device=device)]
    batch = {'class_id': torch.tensor([0, 1, 2]).to(device=device)}
    with pytest.raises(TypeError):
        loss(net_out, batch)


def test_nll_extra_args():
    NLLLoss(reduction='sum', name='name', abc=123)
