import pytest
import torch
from module_list import get_test_module

test_modules = get_test_module()

normal_test = [
    ((2, 16, 5, 5), (2, 32, 5, 5), 16, 32, 1),
    ((1, 10, 10, 10), (1, 20, 10, 10), 10, 20, 1),
    ((5, 16, 20, 20), (5, 32, 20, 20), 16,32, 1),
    ((2, 16, 10, 10), (2, 32, 5, 5), 16,32, 2),
    ((2, 16, 5, 5), (2, 32, 3, 3), 16,32, 2)
]

residual_test = [
    ((2, 16, 5, 5), (2, 16, 5, 5), 16, 16, 1),
    ((1, 17, 10, 10), (1, 17, 10, 10), 17, 17, 1),
    ((2, 16, 5, 5), (2, 16, 3, 3), 16, 16, 2),
    ((1, 17, 10, 10), (1, 17, 5, 5), 17, 17, 2),
]

dense_test = [
    ((2, 16, 5, 5), (2, 48, 5, 5), 16, 32, 1),
    ((1, 10, 10, 10), (1, 30, 10, 10), 10, 20, 1),
    ((5, 16, 20, 20), (5, 48, 20, 20), 16, 32, 1),
    ((2, 16, 10, 10), (2, 48, 5, 5), 16, 32, 2),
    ((2, 16, 5, 5), (2, 48, 3, 3), 16, 32, 2)
]

@pytest.mark.parametrize("test_module", test_modules)
@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", normal_test)
def test_module(input_shape,output_shape,channel_in,channel_out,stride,test_module):
    input = torch.randn(input_shape)
    module = test_module(channel_in,channel_out,stride=stride)
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("test_module", test_modules)
@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", normal_test)
def test_cuda_module(input_shape,output_shape,channel_in,channel_out,stride,test_module):
    input = torch.randn(input_shape).cuda()
    module = test_module(channel_in,channel_out,stride=stride).cuda()
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("test_module", test_modules)
@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", residual_test)
def test_residual_add(input_shape,output_shape,channel_in,channel_out,stride,test_module):
    input = torch.randn(input_shape)
    module = test_module(channel_in, channel_out,stride=stride).to_residual(aggregation="add")
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("test_module", test_modules)
@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", dense_test)
def test_residual_concat(input_shape,output_shape,channel_in,channel_out,stride,test_module):
    input = torch.randn(input_shape)
    module = test_module(channel_in, channel_out,stride=stride).to_residual(aggregation="concatenate")
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape


@pytest.mark.parametrize("test_module", test_modules)
@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", dense_test)
@pytest.mark.parametrize("downsample", ["conv","avg","max"])
def test_dense(input_shape,output_shape,channel_in,channel_out,stride,test_module,downsample):
    input = torch.randn(input_shape)
    module = test_module(channel_in, channel_out,stride=stride).to_dense()
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("test_module", test_modules)
@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", normal_test)
def test_backward(input_shape,output_shape,channel_in,channel_out,stride,test_module):
    input = torch.randn(input_shape,requires_grad=True)
    module = test_module(channel_in,channel_out,stride=stride)
    output = module(input)
    torch.sum(output).backward()
    assert input.grad.shape == input.shape
