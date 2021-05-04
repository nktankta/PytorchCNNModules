import torch
import torch.nn as nn
from .base_module import BaseModule
import torch.nn.functional as F

class InceptionA(BaseModule):
    def __init__(self,in_feature,out_featue,activation=nn.Identity,activation_kwargs={},norm_layer = nn.Identity,**kwargs):
        super(InceptionA,self).__init__(in_feature,out_featue,activation=activation,activation_kwargs=activation_kwargs,norm_layer=norm_layer,**kwargs)

        each_out = out_featue//4
        self.branch_1x1_1 = nn.Conv2d(in_feature, each_out, 1,stride=self.stride)

        self.branch_3x3_1 = nn.Conv2d(in_feature, each_out * 3 // 4, 1)
        self.branch_3x3_2 = nn.Conv2d(each_out * 3 // 4, each_out, 3, stride=self.stride, padding=1)

        self.branch_5x5_1 = nn.Conv2d(in_feature,each_out*3//4,1)
        self.branch_5x5_2 = nn.Conv2d(each_out*3//4,each_out,5,stride=self.stride,padding=2)

        self.branch_pool = nn.Conv2d(in_feature,out_featue-each_out*3,1,stride=1)

        self.act = self.activation(**self.activation_kwargs) if self.activation != nn.Identity else None
        self.norm = self.norm_layer(self.out_feature) if self.norm_layer != nn.Identity else None

    def _forward(self,x):
        branch_1x1 = self.branch_1x1_1(x)
        branch_3x3 = self.branch_3x3_2(self.branch_3x3_1(x))
        branch_5x5 = self.branch_5x5_2(self.branch_5x5_1(x))
        branch_pool = F.avg_pool2d(x,kernel_size=3, stride=self.stride, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        x = torch.cat([branch_1x1,branch_3x3,branch_5x5,branch_pool],axis=1)
        if self.act:
            x = self.act(x)
        if self.norm:
            x = self.norm(x)
        return x

class InceptionB(BaseModule):
    def __init__(self,in_feature,out_featue,activation=nn.Identity,activation_kwargs={},norm_layer = nn.Identity,**kwargs):
        super(InceptionB,self).__init__(in_feature,out_featue,activation=activation,activation_kwargs=activation_kwargs,norm_layer=norm_layer,**kwargs)

        each_out = out_featue//4
        self.branch_1x1_1 = nn.Conv2d(in_feature, each_out, 1,stride=self.stride)

        self.branch_3x3s_1 = nn.Conv2d(in_feature, each_out * 3 // 4, 1)
        self.branch_3x3s_2 = nn.Conv2d(each_out * 3 // 4, each_out, 3, stride=self.stride, padding=1)

        self.branch_3x3d_1 = nn.Conv2d(in_feature,each_out*3//4,1)
        self.branch_3x3d_2 = nn.Conv2d(each_out*3//4,each_out,3,stride=1,padding=1)
        self.branch_3x3d_3 = nn.Conv2d(each_out,each_out,3,stride=self.stride,padding=1)

        self.branch_pool = nn.Conv2d(in_feature,out_featue-each_out*3,1,stride=1)

        self.act = self.activation(**self.activation_kwargs) if self.activation != nn.Identity else None
        self.norm = self.norm_layer(self.out_feature) if self.norm_layer != nn.Identity else None

    def _forward(self,x):
        branch_1x1 = self.branch_1x1_1(x)
        branch_3x3s = self.branch_3x3s_2(self.branch_3x3s_1(x))
        branch_3x3d = self.branch_3x3d_3(self.branch_3x3d_2(self.branch_3x3d_1(x)))
        branch_pool = F.avg_pool2d(x,kernel_size=3, stride=self.stride, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        x = torch.cat([branch_1x1,branch_3x3s,branch_3x3d,branch_pool],axis=1)
        if self.act:
            x = self.act(x)
        if self.norm:
            x = self.norm(x)
        return x

class InceptionC(BaseModule):
    def __init__(self,in_feature,out_featue,n=7,activation=nn.Identity,activation_kwargs={},norm_layer = nn.Identity,**kwargs):
        super(InceptionC,self).__init__(in_feature,out_featue,activation=activation,activation_kwargs=activation_kwargs,norm_layer=norm_layer,**kwargs)

        each_out = out_featue//4
        pad = (n-1)//2
        self.branch_1x1_1 = nn.Conv2d(in_feature, each_out, 1,stride=self.stride)

        self.branch_1xNs_1 = nn.Conv2d(in_feature, each_out * 3 // 4, 1)
        self.branch_1xNs_2 = nn.Conv2d(each_out * 3 // 4, each_out * 3 // 4, kernel_size=(1,n), stride=1, padding=(0,pad))
        self.branch_1xNs_3 = nn.Conv2d(each_out * 3 // 4, each_out, kernel_size=(n,1), stride=self.stride, padding=(pad,0))


        self.branch_1xNd_1 = nn.Conv2d(in_feature,each_out*3//4,1)
        self.branch_1xNd_2 = nn.Conv2d(each_out * 3 // 4, each_out * 3 // 4, kernel_size=(1,n), stride=1, padding=(0,pad))
        self.branch_1xNd_3 = nn.Conv2d(each_out * 3 // 4, each_out * 3 // 4, kernel_size=(n,1), stride=1, padding=(pad,0))
        self.branch_1xNd_4 = nn.Conv2d(each_out * 3 // 4, each_out * 3 // 4, kernel_size=(1,n), stride=1, padding=(0,pad))
        self.branch_1xNd_5 = nn.Conv2d(each_out * 3 // 4, each_out, kernel_size=(n,1), stride=self.stride, padding=(pad,0))

        self.branch_pool = nn.Conv2d(in_feature,out_featue-each_out*3,1,stride=1)

        self.act = self.activation(**self.activation_kwargs) if self.activation != nn.Identity else None
        self.norm = self.norm_layer(self.out_feature) if self.norm_layer != nn.Identity else None

    def _forward(self,x):
        branch_1x1 = self.branch_1x1_1(x)
        branch_1xNs = self.branch_1xNs_3(self.branch_1xNs_2(self.branch_1xNs_1(x)))
        branch_1xNd = self.branch_1xNd_3(self.branch_1xNd_2(self.branch_1xNd_1(x)))
        branch_1xNd = self.branch_1xNd_5(self.branch_1xNd_4(branch_1xNd))
        branch_pool = F.avg_pool2d(x,kernel_size=3, stride=self.stride, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        x = torch.cat([branch_1x1,branch_1xNs,branch_1xNd,branch_pool],axis=1)
        if self.act:
            x = self.act(x)
        if self.norm:
            x = self.norm(x)
        return x

class InceptionD(BaseModule):
    def __init__(self,in_feature,out_featue,activation=nn.Identity,activation_kwargs={},norm_layer = nn.Identity,**kwargs):
        super(InceptionD,self).__init__(in_feature,out_featue,activation=activation,activation_kwargs=activation_kwargs,norm_layer=norm_layer,**kwargs)

        each_out = out_featue//6
        self.branch_1x1_1 = nn.Conv2d(in_feature, each_out, 1,stride=self.stride)

        self.branch_3x3s_1 = nn.Conv2d(in_feature, each_out * 3 // 4, 1)
        self.branch_3x3s_2_1 = nn.Conv2d(each_out * 3 // 4, each_out, kernel_size=(1,3), stride=self.stride, padding=(0,1))
        self.branch_3x3s_2_2 = nn.Conv2d(each_out * 3 // 4, each_out, kernel_size=(3,1), stride=self.stride, padding=(1,0))


        self.branch_3x3d_1 = nn.Conv2d(in_feature,each_out*3//4,1)
        self.branch_3x3d_2 = nn.Conv2d(each_out * 3 // 4, each_out * 3 // 4, kernel_size=3, stride=1, padding=1)
        self.branch_3x3d_3_1 = nn.Conv2d(each_out * 3 // 4, each_out, kernel_size=(1,3), stride=self.stride, padding=(0,1))
        self.branch_3x3d_3_2 = nn.Conv2d(each_out * 3 // 4, each_out, kernel_size=(3,1), stride=self.stride, padding=(1,0))

        self.branch_pool = nn.Conv2d(in_feature,out_featue-each_out*5,1,stride=1)

        self.act = self.activation(**self.activation_kwargs) if self.activation != nn.Identity else None
        self.norm = self.norm_layer(self.out_feature) if self.norm_layer != nn.Identity else None

    def _forward(self,x):
        branch_1x1 = self.branch_1x1_1(x)
        branch_3x3s = self.branch_3x3s_1(x)
        branch_3x3s_1 = self.branch_3x3s_2_1(branch_3x3s)
        branch_3x3s_2 = self.branch_3x3s_2_2(branch_3x3s)
        branch_3x3d = self.branch_3x3d_2(self.branch_3x3d_1(x))
        branch_3x3d_1 =self.branch_3x3d_3_1(branch_3x3d)
        branch_3x3d_2 = self.branch_3x3d_3_2(branch_3x3d)

        branch_pool = F.avg_pool2d(x,kernel_size=3, stride=self.stride, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        x = torch.cat([branch_1x1,branch_3x3s_1,branch_3x3s_2,branch_3x3d_1,branch_3x3d_2,branch_pool],axis=1)
        if self.act:
            x = self.act(x)
        if self.norm:
            x = self.norm(x)
        return x

