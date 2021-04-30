import torch
import torch.nn as nn
import torch.nn.functional as F

class EasyModel(nn.Module):
    def __init__(self,in_channel,classes,test_module,base_module=None,mode="normal",mode_kwargs={}):
        super(EasyModel,self).__init__()
        self.test_module = test_module
        self.base_module = base_module
        self.in_channel = in_channel
        self.output_classes = classes
        self.conv = nn.Conv2d(in_channel,16,1,1,0)
        self.linear = nn.Linear(16 * 2**4,self.output_classes)
        self.hidden_layer_num = [2,2,2,2]
        self.mode_kwargs = mode_kwargs
        self.layer1,self.layer2,self.layer3,self.layer4 = [self.make_layer(16*(2**i),self.hidden_layer_num[i],mode) for i in range(len(self.hidden_layer_num))]
        self.layers = [self.layer1,self.layer2,self.layer3,self.layer4]

    def make_layer(self,hidden_layer,repeat,mode="residual"):
        if mode=="residual" or mode=="normal":
            return self._make_layer_normal(hidden_layer,repeat,mode=mode)
        elif mode=="dense":
            return self._make_layer_dense(hidden_layer,repeat,mode=mode)

    def _make_layer_normal(self,hidden_layer,repeat,mode="residual"):
        base_module = self.test_module if self.base_module is None else self.base_module
        extra_module = self.test_module if self.base_module is not None else None
        layer_list = []
        for i in range(repeat):
            if i!=repeat-1:
                layer_list.append(base_module(hidden_layer,hidden_layer).change_mode(mode,**self.mode_kwargs))
                if extra_module is not None:
                    layer_list.append(extra_module(hidden_layer, hidden_layer).change_mode(mode,**self.mode_kwargs))
            else:
                layer_list.append(base_module(hidden_layer, hidden_layer*2,stride=2).change_mode(mode,**self.mode_kwargs))
        return nn.Sequential(*layer_list)

    def _make_layer_dense(self,hidden_layer,repeat,mode):
        base_module = self.test_module if self.base_module is None else self.base_module
        extra_module = self.test_module if self.base_module is not None else None
        layer_list = []
        input_ch = hidden_layer
        for i in range(repeat):
            layer_list.append(base_module(input_ch,2**(i+1)).change_mode(mode))
            input_ch += 2**(i+1)
            if extra_module is not None:
                layer_list.append(extra_module(input_ch, input_ch).change_mode(mode))
        layer_list.append(nn.Conv2d(input_ch,hidden_layer*2,1,2,0))
        return nn.Sequential(*layer_list)
    def forward(self,x):
        x = self.conv(x)
        for i in self.layers:
            x = i(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view((-1,256))
        x = torch.sigmoid(self.linear(x))
        return x

if __name__ == '__main__':
    from PytorchCnnModules import *
    model = EasyModel(1,10,InvertedResidual,mode="dense").cuda()
    inp = torch.randn((5,1,28,28)).cuda()
    out = model(inp)
    print(out.shape)