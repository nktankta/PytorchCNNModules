import pytest
import torch
import torch.nn as nn
from module_list import get_test_module
from PytorchCnnModules.modules.base_module import BaseModule,SEmodule

class CNN(BaseModule):
    def __init__(self,in_feature,out_featue,stride=1):
        super(CNN,self).__init__(in_feature,out_featue,stride)
        self.cnn = nn.Conv2d(in_feature,out_featue,3,stride,1)
    def _forward(self,x):
        return self.cnn(x)

class Identity(BaseModule):
    def __init__(self,in_feature,out_featue,stride=1,**kwargs):
        super(Identity,self).__init__(in_feature,out_featue,stride,norm_layer = nn.Identity,**kwargs)
    def _forward(self,x,*args):
        if self.stride!=1:
            x = nn.AvgPool2d(1,stride=self.stride)(x)
        return x

test_modules = get_test_module()

normal_test = [
    ((2, 3, 5, 5), (2, 10, 5, 5), 3, 10, 1),
    ((1, 10, 10, 10), (1, 20, 10, 10), 10, 20, 1),
    ((5, 8, 20, 20), (5, 16, 20, 20), 8, 16, 1),
    ((2, 3, 10, 10), (2, 10, 5, 5), 3, 10, 2),
    ((2, 3, 5, 5), (2, 10, 3, 3), 3, 10, 2)
]

residual_test = [
    ((2, 10, 5, 5), (2, 10, 5, 5), 10, 10, 1),
    ((1, 17, 10, 10), (1, 17, 10, 10), 17, 17, 1),
    ((2, 10, 5, 5), (2, 10, 3, 3), 10, 10, 2),
    ((1, 17, 10, 10), (1, 17, 5, 5), 17, 17, 2),
]

dense_test = [
    ((2, 3, 5, 5), (2, 13, 5, 5), 3, 10, 1),
    ((1, 10, 10, 10), (1, 30, 10, 10), 10, 20, 1),
    ((5, 8, 20, 20), (5, 24, 20, 20), 8, 16, 1),
    ((2, 3, 10, 10), (2, 13, 5, 5), 3, 10, 2),
    ((2, 3, 5, 5), (2, 13, 3, 3), 3, 10, 2)
]




def test_residual_featuresize_exception():
    with pytest.raises(AssertionError,match="[residual feature size error]"):
        CNN(10,5,1).to_residual()

def test_residual_aggregation_error():
    with pytest.raises(NotImplementedError):
        CNN(10,10,2).to_residual(aggregation="test")

def test_dense_downsample_error():
    with pytest.raises(NotImplementedError):
        CNN(10, 10, 2).to_dense(downsample="test")

def test_residual_activation_bool():
    inp = torch.randn((2,10,20,20))
    module = Identity(10,10,1).to_residual(activation=True)
    out = module(inp)
    assert torch.min(out).item()>=0

def test_residual_activation():
    inp = torch.ones((2,10,20,20))*10
    module = Identity(10,10,1).to_residual(activation=nn.ReLU6)
    out = module(inp)
    assert torch.max(out).item()<=6

def test_residual_preactivation_bool():
    inp = -torch.ones((2,10,20,20))
    module = Identity(10,10,1).to_residual(pre_activation=True)
    out = module(inp)
    assert torch.min(out).item()>=-1

def test_residual_preactivation():
    inp = torch.ones((2,10,20,20))*10
    module = Identity(10,10,1).to_residual(pre_activation=nn.ReLU6)
    out = module(inp)
    assert torch.max(out).item()<=16


def test_residual_random_drop():
    inp = torch.ones((2,10,20,20))*1
    module = Identity(10,10,1).to_residual(drop_rate=1)
    out = module(inp)
    assert torch.max(out).item()<=1


def test_semodule():
    inp = torch.randn((5, 31, 11, 11))
    module = SEmodule(31)
    out = module(inp)
    assert out.shape == inp.shape


def test_semodule_enable():
    inp = torch.ones((2,10,20,20))
    out_normal = torch.empty((2,10,20,20))
    out_dense = torch.empty((2,20,20,20))
    module = Identity(10,10,1,use_SEmodule=True)
    out = module(inp)
    assert out.shape == out_normal.shape
    module.to_residual()
    out = module(inp)
    assert out.shape == out_normal.shape
    module.to_dense()
    out = module(inp)
    assert out.shape == out_dense.shape

def test_multi_input():
    inp1 = torch.randn((2,10,20,20))
    inp2 = torch.randn((2,10,20,20))
    dense_out = torch.empty((2,20,20,20))
    module = Identity(10,10,1)
    out = module(inp1,inp2)
    assert out.shape == inp1.shape
    module.to_residual()
    out = module(inp1,inp2)
    assert out.shape == inp1.shape
    module.to_dense()
    out = module(inp1,inp2)
    assert out.shape == dense_out.shape

@pytest.mark.parametrize("input_shape", [(2,10,20,20),(2,10,5,5)])
@pytest.mark.parametrize("output_feature", [10,20])
@pytest.mark.parametrize("downsample", ["conv","max","avg"])
def test_residual_downsample_add(input_shape,output_feature,downsample):
    n,c,w,h = input_shape
    inp = torch.randn(input_shape)
    downsample_out = (n,output_feature,(w-1)//2+1,(h-1)//2+1)
    module = CNN(10,output_feature,2).to_residual(aggregation="add",downsample=downsample)
    out = module(inp)
    assert out.shape == torch.empty(downsample_out).shape

@pytest.mark.parametrize("input_shape,output_shape", [((2,10,20,20),(2,20,10,10)),((2,20,5,5),(2,40,3,3))])
@pytest.mark.parametrize("downsample", ["conv","max","avg"])
def test_residual_downsample_conc(input_shape,output_shape,downsample):
    inp = torch.randn(input_shape)
    module = CNN(input_shape[1],input_shape[1],2).to_residual(aggregation="concatenate",downsample=downsample)
    out = module(inp)
    assert out.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("input_shape,output_shape", [((2,10,20,20),(2,20,10,10)),((2,20,5,5),(2,40,3,3))])
@pytest.mark.parametrize("downsample", ["conv","max","avg"])
def test_dense_downsample(input_shape,output_shape,downsample):
    inp = torch.randn(input_shape)
    module = CNN(input_shape[1],input_shape[1],2).to_dense(downsample)
    out = module(inp)
    assert out.shape == torch.empty(output_shape).shape




@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", normal_test)
def test_module(input_shape,output_shape,channel_in,channel_out,stride):
    input = torch.randn(input_shape)
    module = CNN(channel_in,channel_out,stride)
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", normal_test)
def test_cuda_module(input_shape,output_shape,channel_in,channel_out,stride):
    input = torch.randn(input_shape).cuda()
    module = CNN(channel_in,channel_out,stride).cuda()
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", residual_test)
def test_residual_add(input_shape,output_shape,channel_in,channel_out,stride):
    input = torch.randn(input_shape)
    module = CNN(channel_in, channel_out, stride).to_residual(aggregation="add")
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", dense_test)
def test_residual_concat(input_shape,output_shape,channel_in,channel_out,stride):
    input = torch.randn(input_shape)
    module = CNN(channel_in, channel_out, stride).to_residual(aggregation="concatenate")
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape



@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", dense_test)
@pytest.mark.parametrize("downsample", ["conv","avg","max"])
def test_dense(input_shape,output_shape,channel_in,channel_out,stride,downsample):
    input = torch.randn(input_shape)
    module = CNN(channel_in, channel_out, stride).to_dense()
    output = module(input)
    assert output.shape == torch.empty(output_shape).shape

@pytest.mark.parametrize("input_shape,output_shape,channel_in,channel_out,stride", normal_test)
def test_backward(input_shape,output_shape,channel_in,channel_out,stride):
    input = torch.randn(input_shape,requires_grad=True)
    module = CNN(channel_in,channel_out,stride)
    output = module(input)
    torch.sum(output).backward()
    assert input.grad.shape == input.shape