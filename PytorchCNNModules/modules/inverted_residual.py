import torch
import torch.nn as nn
from .base_module import BaseModule

class InvertedResidual(BaseModule):
    def __init__(self,in_feature,out_featue,stride=1,activation=nn.ReLU6,activation_kwargs={"inplace":True},t=2,**kwargs):
        super(InvertedResidual,self).__init__(in_feature,out_featue,stride,**kwargs)
        self.conv1 = nn.Conv2d(in_feature,in_feature*t,1,1,padding=0)
        self.conv2 = nn.Conv2d(in_feature*t,in_feature*t,3,stride,padding=1,groups=in_feature*t)
        self.conv3 = nn.Conv2d(in_feature*t,out_featue,1,1,padding=0)
        self.act1 = activation(**activation_kwargs)
        self.act2 = activation(**activation_kwargs)
        self.norm1 = self.norm_layer(in_feature*t)
        self.norm2 = self.norm_layer(in_feature*t)

    def _forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x