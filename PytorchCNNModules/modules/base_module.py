import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SEmodule(nn.Module):
    def __init__(self,in_feature,ratio=8,activation=nn.ReLU,activation_kwargs={"inplace":True}):
        super(SEmodule,self).__init__()
        self.feature = in_feature
        self.linear1 = nn.Linear(in_feature,in_feature//ratio)
        self.linear2 = nn.Linear(in_feature//ratio,in_feature)
        self.act1 = activation(**activation_kwargs)

    def forward(self,x):
        inp = x
        pool = F.adaptive_avg_pool2d(x, (1, 1)).view((-1,self.feature))
        x = self.act1(self.linear1(pool))
        x = torch.sigmoid(self.linear2(x))
        att = x.unsqueeze(-1).unsqueeze(-1)
        return inp * att


class BaseModule(nn.Module):
    def __init__(self,in_feature,out_featue,stride=1,use_SEmodule=False,SEmodule_ratio=8,norm_layer=nn.BatchNorm2d,activation=nn.ReLU,activation_kwargs={"inplace":True}):
        super(BaseModule,self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_featue
        self.stride = stride
        self.norm_layer = norm_layer
        self.mode = "normal"
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.residual_activation = None
        self.pre_activation = None
        self.use_SEmodule = use_SEmodule
        self.SEmodule_ratio = SEmodule_ratio
        if self.use_SEmodule:
            self.SEmodule = SEmodule(self.out_feature,self.SEmodule_ratio)

    def change_mode(self,mode,**kwargs):
        if mode == "normal":
            self.to_normal(**kwargs)
        if mode == "residual":
            self.to_residual(**kwargs)
        if mode == "dense":
            self.to_dense(**kwargs)
        return self

    def to_normal(self):
        self.mode = "normal"
        return self

    def get_downsampler(self,downsample):
        if downsample == "conv":
            return nn.Conv2d(self.in_feature, self.in_feature, 1, self.stride, 0)
        elif downsample == "avg":
            return nn.AvgPool2d((3,3),stride = self.stride,padding=1)
        elif downsample == "max":
            return nn.MaxPool2d((3,3),stride =self.stride, padding=1)
        else:
            raise NotImplementedError
        
    def to_residual(self,aggregation="add",drop_rate=0,downsample="conv",pre_activation=False,activation=True,activation_kwargs={}):
        self.residual_drop = drop_rate
        if activation:
            if activation == True:
                activation = self.activation
                activation_kwargs = self.activation_kwargs
            self.residual_activation = activation(**activation_kwargs)
        if pre_activation:
            if pre_activation==True:
                pre_activation = self.activation
                activation_kwargs = self.activation_kwargs
            self.pre_bn = self.norm_layer(self.in_feature)
            self.pre_activation = pre_activation(**activation_kwargs)

        if aggregation == "add":
            self._to_residual_add(downsample)
        elif aggregation == "concatenate":
            self._to_residual_concatenate(downsample)
        else:
            raise NotImplementedError

        self.mode = "residual"
        self.agg = aggregation
        return self

    def _to_residual_add(self,downsample):
        if self.stride ==1 & (self.in_feature != self.out_feature):
            raise AssertionError(f"residual feature size error "
                                 f"input and output feature size must have the same channel."
                                 f"but input :{self.in_feature} output:{self.out_feature}")
        if self.stride!=1:
            if self.in_feature != self.out_feature:
                self.downsample = nn.Conv2d(self.in_feature, self.out_feature, 1, self.stride, 0)
            else:
                self.downsample = self.get_downsampler(downsample)

    def _to_residual_concatenate(self,downsample):
        if self.use_SEmodule:
            self.SEmodule = SEmodule(self.in_feature + self.out_feature, self.SEmodule_ratio)
        if self.stride!=1:
            self.downsample = self.get_downsampler(downsample)

    def to_dense(self,downsample="conv"):
        if self.use_SEmodule:
            self.SEmodule = SEmodule(self.in_feature + self.out_feature, self.SEmodule_ratio)
        if self.stride != 1:
            self.downsample = self.get_downsampler(downsample)
        self.mode = "dense"
        return self

    def forward(self,x,*args):
        if self.mode == "normal":
            out =  self._forward(x,*args)
        elif self.mode == "residual":
            out =  self._residual_forward(x,*args)
        elif self.mode == "dense":
            out = self._dense_forward(x,*args)
        else:
            raise NotImplementedError
        if self.use_SEmodule:
            out = self.SEmodule(out)
        return out

    def _forward(self,x):
        raise NotImplementedError

    def _residual_forward(self,x,*args):
        if self.residual_drop:
            if random.random()<self.residual_drop:
                return x
        if self.pre_activation:
            x = self.pre_activation(self.pre_bn(x))
        out = self._forward(x,*args)
        if self.stride !=1:
            x = self.downsample(x)
        if self.agg == "add":
            output =  out+x
        elif self.agg == "concatenate":
            output =  torch.cat([x,out],axis = 1)
        else:
            NotImplementedError
        if self.residual_activation:
            output = self.residual_activation(output)
        return output

    def _dense_forward(self,x,*args):
        out = self._forward(x,*args)
        if self.stride!=1:
            x = self.downsample(x)
        return torch.cat([x,out],axis = 1)

