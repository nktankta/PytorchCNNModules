import torch
import torch.nn as nn
from PytorchCNNModules.modules.base_module import BaseModule
from PytorchCNNModules import *

class CNN(BaseModule):
    def __init__(self,in_feature,out_featue,stride=1):
        super(CNN,self).__init__(in_feature,out_featue,stride)
        self.cnn = nn.Conv2d(in_feature,out_featue,3,stride,1)
        self.norm = self.norm_layer(out_featue)
        self.act = self.activation(**self.activation_kwargs)
    def _forward(self,x):
        return self.act(self.norm(self.cnn(x)))

test_module = [InceptionD]
def get_test_module():
    return test_module

# passed
# InvertedResidual,PlaneResidual,PlaneResidual_no_lastBN,BottleneckResidual,BottleneckResidual_no_lastBN