# -*- coding: utf-8 -*-
"""
@Time ：　2021/1/14
@Auth ：　xiaer.wang
@File ：　clustering.py
@IDE 　：　PyCharm
"""
from torch.nn import functional as F
import models
from modules.pairwise import PairEnum
from .template import Template


class Learner_Clustering(Template):

    @staticmethod
    def create_model(model_type,model_name,in_dim, out_dim):
        # Prepare Constrained Clustering Network (CCN)
        model = models.__dict__[model_type].__dict__[model_name](in_dim=in_dim, out_dim=out_dim)
        return model

    def forward(self, x):
        logits = self.model.forward(x)
        prob = F.softmax(logits,dim=1)
        return prob

    def forward_with_criterion(self, inputs, simi, mask=None, **kwargs):
        prob = self.forward(inputs)
        prob1, prob2 = PairEnum(prob, mask)
        return self.criterion(prob1, prob2, simi),prob
