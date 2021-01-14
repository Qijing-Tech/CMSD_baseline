# -*- coding: utf-8 -*-
"""
@Time ：　2021/1/12
@Auth ：　xiaer.wang
@File ：　l2c.py
@IDE 　：　PyCharm
"""
import torch.nn as nn

class L2C(nn.Module):

    def __init__(self,n_layer,  in_dim, out_dim):
        super(L2C, self).__init__()
        self.n_layer = n_layer
        if n_layer == 0:
            # due to provide embedding, set None
            self.features = None
            self.last_feature = in_dim
        else:
            dim_list = [in_dim // pow(2,i) for i in range(1, n_layer+1)]

            self.features = self._make_layers(dim_list, in_dim)
            self.last_feature = dim_list[-1]

        self.last = nn.Linear( self.last_feature, out_dim)

    def _make_layers(self, dim_list, in_dim):
        layers = []
        for x in dim_list:

            layers += [
                nn.Linear(in_features= in_dim, out_features=x),
                nn.Tanh()
            ]
            in_dim = x

        return nn.Sequential(*layers)

    def logits(self, x):
        x = self.last(x)
        return x
    def forward(self, x):
        if self.n_layer != 0:
            x = self.features(x)
        x = self.logits(x.view(x.size(0), -1))
        return x

def L2C_v0(in_dim, out_dim):
    return L2C(n_layer = 0, in_dim=in_dim, out_dim=out_dim)

def L2C_v1(in_dim, out_dim):
    return L2C(n_layer = 1, in_dim=in_dim, out_dim=out_dim)

def L2C_v2(in_dim, out_dim):
    return L2C(n_layer=2, in_dim=in_dim, out_dim=out_dim)

def L2C_v4(in_dim, out_dim):
    return L2C(n_layer=4, in_dim=in_dim, out_dim=out_dim)

def L2C_v8(in_dim, out_dim):
    return L2C(n_layer=8, in_dim=in_dim, out_dim=out_dim)
