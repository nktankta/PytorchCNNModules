import torch
import torch.nn as nn
from .base_module import BaseModule

class PlaneResidual(BaseModule):
    def __init__(self,in_feature,out_featue,hidden_feature=None,stride=1,**kwargs):
        super(PlaneResidual,self).__init__(in_feature,out_featue,stride,**kwargs)
        if hidden_feature is None:
            hidden_feature = in_feature
        self.conv1 = nn.Conv2d(in_feature,hidden_feature,3,stride,padding=1)
        self.conv2 = nn.Conv2d(hidden_feature,out_featue,3,1,padding=1)
        self.act1 = self.activation(**self.activation_kwargs)
        self.norm1 = self.norm_layer(hidden_feature)
        self.norm2 = self.norm_layer(out_featue)

    def _forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class PlaneResidual_no_lastBN(BaseModule):
    def __init__(self, in_feature, out_featue, hidden_feature=None, stride=1, **kwargs):
        super(PlaneResidual_no_lastBN, self).__init__(in_feature, out_featue, stride, **kwargs)
        if hidden_feature is None:
            hidden_feature = in_feature
        self.conv1 = nn.Conv2d(in_feature, hidden_feature, 3, stride, padding=1)
        self.conv2 = nn.Conv2d(hidden_feature, out_featue, 3, 1, padding=1)
        self.act1 = self.activation(**self.activation_kwargs)
        self.norm1 = self.norm_layer(hidden_feature)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x
