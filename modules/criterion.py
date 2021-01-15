# -*- coding: utf-8 -*-
"""
@Time ：　2021/1/14
@Auth ：　xiaer.wang
@File ：　criterion.py
@IDE 　：　PyCharm
"""
import torch.nn as nn

eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).

class KLDiv(nn.Module):
    # Calculate KL-Divergence

    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict += eps
        target += eps
        logI = predict.log()
        logT = target.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld

class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL, self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),
                                                                                            str(len(prob2)),
                                                                                            str(len(simi)))

        kld = self.kld(prob1, prob2)
        output = self.hingeloss(kld, simi)
        return output


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),
                                                                                            str(len(prob2)),
                                                                                            str(len(simi)))

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()
        return neglogP.mean()
