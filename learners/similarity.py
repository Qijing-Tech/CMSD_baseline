# -*- coding: utf-8 -*-
"""
@Time ：　2021/1/14
@Auth ：　xiaer.wang
@File ：　similarity.py
@IDE 　：　PyCharm
"""
import models
import torch
import torch.nn as nn
from types import MethodType
from .template import Template
from modules.pairwise import PairEnum

class Learner_PairSimilarity(Template):

    @staticmethod
    def create_model(model_type, model_name, in_dim, out_dim):
        model = models.__dict__[model_type].__dict__[model_name](in_dim=in_dim, out_dim=out_dim)

        n_feat = model.last.in_features

        #Replace task-dependent module
        model.last = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat * 4),
            nn.Tanh(),
            nn.Linear(n_feat * 4, 2)
        )

        # Replace task-dependent function
        def new_logits(self, x):
            feat1, feat2 = PairEnum(x)
            featcat = torch.cat([feat1, feat2], 1)
            out = self.last(featcat)
            return out

        model.logits = MethodType(new_logits, model)

        return model