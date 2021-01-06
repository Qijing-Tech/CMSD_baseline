# -*- coding: utf-8 -*-
"""
@Time ：　2020/12/31
@Auth ：　xiaer.wang
@File ：　method.py
@IDE 　：　PyCharm
"""
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
'''
    wrapper from sklearn
'''
class Kmeans:
    def __init__(self, n_cluster, seed = 1234):
        self.n_cluster = n_cluster
        self.seed = seed

    def predict(self, data):

        kmeans = KMeans(n_clusters=self.n_cluster, random_state=self.seed).fit(data)

        return kmeans.labels_

class DBScan:
    def __init__(self, eps, min_sample):
        self.eps = eps
        self.min_sample = min_sample

    def predict(self, data):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(data)
        return dbscan.labels_

class GMMs:
    def __init__(self, n_component, seed = 1234):
        self.n_component = n_component
        self.seed = seed

    def predict(self, data):
        gmms = GaussianMixture(n_components=self.n_component, random_state=self.seed).fit(data)
        return gmms.predict(data)