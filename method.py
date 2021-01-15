# -*- coding: utf-8 -*-
"""
@Time ：　2020/12/31
@Auth ：　xiaer.wang
@File ：　method.py
@IDE 　：　PyCharm
"""
import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,OPTICS
from sklearn.mixture import GaussianMixture
'''
    wrapper from sklearn
'''
import networkx as nx
import random
from typing import List
import torch
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import community
from learners.similarity import  Learner_PairSimilarity
from learners.clustering import Learner_Clustering
import modules
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR
from modules.metric import Confusion,Timer,AverageMeter
from modules.pairwise import Class2Simi
cwd = Path.cwd()

class Kmeans:
    def __init__(self, n_cluster, seed = 1234):
        self.n_cluster = n_cluster
        self.seed = seed

    def predict(self, data):

        kmeans = KMeans(n_clusters=self.n_cluster, random_state=self.seed).fit(data)

        return kmeans.labels_

class AC:
    def __init__(self,n_cluster):
        self.n_cluster = n_cluster

    def predict(self, data):
        ac = AgglomerativeClustering(n_clusters= self.n_cluster).fit(data)

        return ac.labels_

class DBScan:
    def __init__(self, eps, min_sample):
        self.eps = eps
        self.min_sample = min_sample

    def predict(self, data):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit(data)
        return dbscan.labels_

class Optics:
    def __init__(self, min_sample):
        self.min_sample = min_sample

    def predict(self, data):
        optics = OPTICS(min_samples=self.min_sample).fit(data)
        return optics.labels_

class GMMs:
    def __init__(self, n_component, seed = 1234):
        self.n_component = n_component
        self.seed = seed

    def predict(self, data):
        gmms = GaussianMixture(n_components=self.n_component, random_state=self.seed).fit(data)
        return gmms.predict(data)

class Louvain:
    def __init__(self, word_list, measure = 'cosine', thresold=0.):
        self.word_list = word_list
        self.map_idx = {i: word for i, word in enumerate(word_list)}
        self.measure = measure
        self.thresold=  thresold

    def predict(self, data):

        graph = self._construct_graph_by_words(self.word_list, data, self.measure, self.thresold)
        partition = community.best_partition(graph)
        pred_labels = [partition[word] if word in partition.keys() else -1 for idx, word in self.map_idx.items()]
        return pred_labels

    def _construct_graph_by_words(self,word_lists : List, word_embeddings : np.ndarray, measure='cosine', thresold=0.):

        #有个问题 ： weight 用什么衡量最好 ？
        def get_distance(v1, v2, measure='cosine'):
            if measure == 'euclidean':
                return np.linalg.norm(v1 - v2)
            elif measure == 'cosine':
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif measure == 'dot':
                return np.dot(v1, v2)
            else:
                raise NotImplementedError

        g = nx.Graph()
        for i, word_prev in enumerate(word_lists):
            for j, word_after in enumerate(word_lists):
                if i < j:
                    weight = get_distance(word_embeddings[i], word_embeddings[j], measure = measure)

                    if weight >= thresold:
                        g.add_edge(word_prev, word_after, weight= weight)


        return g


class L2C:
    def __init__(self, args, in_dim, out_dim, SPN_model = None):

        self.args = args
        self.in_dim = in_dim

        if args.loss_type in ['KCL', 'MCL']:
            LearnerClass = Learner_Clustering
            self.criterion = modules.criterion.__dict__[args.loss_type]()
            self.out_dim = out_dim
        elif args.loss_type == 'DPS':
            LearnerClass = Learner_PairSimilarity
            self.criterion = nn.CrossEntropyLoss()
            self.out_dim = 2 #force
        else:
            raise NotImplementedError

        self.model = LearnerClass.create_model(model_type=args.model_type, model_name=args.model_name,
                                               in_dim=in_dim, out_dim=out_dim)

        if SPN_model:
            SPN = Learner_PairSimilarity.create_model(args.SPN_model_type, args.SPN_model_name, in_dim, 2)
            print('=> Load SPN model weights:',SPN_model)
            SPN_state = torch.load(SPN_model,
                                   map_location=lambda storage, loc: storage)  # Load to CPU as the default!
            SPN.load_state_dict(SPN_state)
            print('=> Load SPN Done')
            args.SPN = SPN

        if args.use_gpu:
            torch.cuda.set_device(args.gpuid[0])
            cudnn.benchmark = True  # make it train faster
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            if args.SPN is not None:
                args.SPN = args.SPN.cuda()

            # Prepare the learner
        optim_args = {'lr': args.lr}
        if args.optimizer == 'SGD':
            optim_args['momentum'] = 0.9
        optimizer = torch.optim.__dict__[args.optimizer](self.model.parameters(), **optim_args)
        scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
        self.learner = LearnerClass(self.model, self.criterion, optimizer, scheduler)


    def train(self, args, train_loader, eval_loader, tgt_class):

        for epoch in range(args.start_epoch, args.epochs):
            self._train_epoch(epoch, train_loader, self.learner, args, tgt_class)
            if eval_loader is not None and ((not args.skip_eval) or (epoch == args.epochs - 1)):
                KPI, _ = self.evaluate(eval_loader, self.model, args)
            # Save checkpoint at each LR steps and the end of optimization
            if (epoch + 1 in args.schedule + [args.epochs]) and not args.skip_eval:
                self.learner.snapshot(os.path.join(args.model_save_path, "%s#%s#%s#%s" % (args.data, args.embed, args.model_name, args.loss_type)), KPI)
        return KPI

    def evaluate(self, eval_loader, model, args, tgt_class = None):

        # Initialize all meters
        confusion = Confusion(args.out_dim)

        print('---- Evaluation ----')
        model.eval()
        for i, (input, target) in enumerate(eval_loader):

            # Prepare the inputs
            if args.use_gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            _, eval_target = self._prepare_task_target(input, target, args)

            # Inference
            output = model(input)

            # Update the performance meter
            output = output.detach()

            confusion.add(output, eval_target)

        # Loss-specific information
        KPI = 0
        cluster_index = None
        if args.loss_type in ['KCL', 'MCL']:

            confusion.optimal_assignment(tgt_class, args.cluster2Class)
            if args.out_dim <= 20:
                confusion.show()
            cluster_index = confusion.clusterscores()
            print('Clustering scores:', confusion.clusterscores())
            KPI = confusion.acc()
            print('[Test] ACC: ', KPI)
        elif args.loss_type == 'DPS':
            confusion.show(width=15, row_labels=['GT_dis-simi', 'GT_simi'],
                           column_labels=['Pred_dis-simi', 'Pred_simi'])
            KPI = confusion.f1score(1)
            print('[Test] similar pair f1-score:', KPI)  # f1-score for similar pair (label:1)
            print('[Test] dissimilar pair f1-score:', confusion.f1score(0))
        return KPI, cluster_index

    def _prepare_task_target(self, input, target, args):
        # Prepare the target for different criterion/tasks
        if args.loss_type in ['KCL', 'MCL']:  # For clustering
            if args.use_SPN:  # For unsupervised clustering
                # Feed the input to SPN to get predictions
                _, train_target = args.SPN(input).max(1)  # Binaries the predictions
                train_target = train_target.float()
                train_target[train_target==0] = -1  # Simi:1, Dissimi:-1
            else:  # For supervised clustering
                # Convert class labels to pairwise similarity
                train_target = Class2Simi(target, mode='hinge')
            eval_target = target
        elif args.loss_type == 'DPS':  # For learning the SPN
            train_target = eval_target = Class2Simi(target, mode='cls')
        else:
            assert False,'Unsupported loss:'+args.loss_type

        return train_target.detach(), eval_target.detach()  # Make sure no gradients

    def _train_epoch(self, epoch, train_loader, learner, args, tgt_class):
        # This function optimize the objective

        # Initialize all meters
        data_timer = Timer()
        batch_timer = Timer()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        confusion = Confusion(args.out_dim)

        # Setup learner's configuration
        print('\n\n==== Epoch:{0} ===='.format(epoch))
        learner.train()
        learner.step_schedule(epoch)

        # The optimization loop
        data_timer.tic()
        batch_timer.tic()
        if args.print_freq > 0:  # Enable to print mini-log
            print('Itr            |Batch time     |Data Time      |Loss')
        for i, (input, target) in enumerate(train_loader):

            data_time.update(data_timer.toc())  # measure data loading time
            # Prepare the inputs
            if args.use_gpu:
                input = input.cuda()
                target = target.cuda()
            train_target, eval_target = self._prepare_task_target(input, target, args)

            # Optimization
            loss, output = learner.learn(input, train_target)

            # Update the performance meter
            confusion.add(output, eval_target)

            # Measure elapsed time
            batch_time.update(batch_timer.toc())
            data_timer.toc()

            # Mini-Logs
            losses.update(loss, input.size(0))
            if args.print_freq > 0 and ((i % args.print_freq == 0) or (i == len(train_loader) - 1)):
                print('[{0:6d}/{1:6d}]\t'
                      '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                      '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                      '{loss.val:.3f} ({loss.avg:.3f})'.format(
                    i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        # Loss-specific information
        if args.loss_type in ['KCL', 'MCL']:
            args.cluster2Class = confusion.optimal_assignment(tgt_class)  # Save the mapping in args to use in eval
            if args.out_dim <= 20:  # Avoid to print a large confusion matrix
                confusion.show()
            print('Clustering scores:', confusion.clusterscores())
            print('[Train] ACC: ', confusion.acc())
        elif args.loss_type == 'DPS':
            confusion.show(width=15, row_labels=['GT_dis-simi', 'GT_simi'],
                           column_labels=['Pred_dis-simi', 'Pred_simi'])
            print('[Train] similar pair f1-score:', confusion.f1score(1))  # f1-score for similar pair (label:1)
            print('[Train] dissimilar pair f1-score:', confusion.f1score(0))

    def predict(self, test_loader, args, tgt_class):

        _, cluster_info = self.evaluate(test_loader,self.model,args, tgt_class)
        return cluster_info