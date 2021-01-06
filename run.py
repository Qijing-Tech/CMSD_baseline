# -*- coding: utf-8 -*-
"""
@Time ：　2020/12/31
@Auth ：　xiaer.wang
@File ：　run_test.py
@IDE 　：　PyCharm
"""
from pathlib import Path
from dataloader import DataSet,DataSetDir
from method import Kmeans,GMMs,DBScan,AC
import numpy as np
import random
from typing import List,Tuple,Dict
from collections import OrderedDict
from evaluate import select_evaluate_func,metrics_adjusted_randn_index,\
    metrics_normalized_mutual_info_score,metrics_fowlkes_mallows_score
from config import DataConfig, DATA_ROOT
from utils import set_random_seed
from logger import Logger
from utils import split_sub_train_set_by_dev_set
from args import parser
args = parser.parse_args()
cwd = Path.cwd()
SEED = 2020


def get_word_idxes_and_cluster_idxes(raw_word_list : List[List[str]], vocab_dict : Dict,
                                     word2id : Dict,  is_shuffle = True) -> Tuple[List, List, Dict]:
    flat_word_list = []
    for word_list in raw_word_list:
        flat_word_list.extend(word_list)
    if is_shuffle:
        random.shuffle(flat_word_list)

    word_to_idxes = []
    for word in flat_word_list:
        word_to_idxes.append(word2id[word])

    cluster_idxes = OrderedDict()
    idx = 0
    for word, cluster in vocab_dict.items():
        if cluster not in cluster_idxes:
            cluster_idxes[cluster] = idx
            idx += 1

    return flat_word_list, word_to_idxes, cluster_idxes

def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

def format_output(data, bit = 1):
    if isinstance(data, float):
        return round(data * 100, bit)
    elif isinstance(data, list):
       return [round(d * 100, bit)  for d in data]


if __name__ == '__main__':
    DataConfig['data_name'] = args.data
    DataConfig['method_type'] = args.method
    #DataConfig['word_emb_select'] = args.embed
    set_random_seed(seed = SEED)
    datadir_name = DataConfig['data_name']
    data = DATA_ROOT.joinpath(datadir_name)
    #TODO : need modify (word_emb_select) -> choice ['combined.embed', 'Tencent_combined.embed']
    word_emb_select = DataConfig['word_emb_select']
    method_type = DataConfig['method_type']

    datadir = DataSetDir(data,word_emb_select = word_emb_select)
    word2id = datadir.word2id
    embedding = datadir.embedding_vec

    train_vocab = datadir.train_dataset.vocab
    train_sets = datadir.train_dataset.raw_sets

    dev_vocab = datadir.dev_dataset.vocab
    dev_sets = datadir.dev_dataset.raw_sets

    # # for train
    # train_flat_word_list, train_word_idxes, train_cluster_idxes = get_word_idxes_and_cluster_idxes(train_sets, train_vocab, word2id)
    # train_word_embeddings = embedding[np.array(train_word_idxes)]

    dev_flat_word_list, dev_word_idxes, dev_cluster_idxes = get_word_idxes_and_cluster_idxes(dev_sets, dev_vocab, word2id)
    dev_word_embeddings = embedding[np.array(dev_word_idxes)]

    #TODO : need modify (run_times)
    run_times = DataConfig['run_times']
    seed_list = DataConfig['seed_list']
    ari_list, nmi_list, fmi_list = [], [], []
    for i in range(run_times):
        # TODO : model
        if method_type == 'kmeans':
            # set cluster number by prior
            k_cluster = len(dev_cluster_idxes.keys())
            model = Kmeans(n_cluster=k_cluster, seed=seed_list[i])
            pred_labels = model.predict(dev_word_embeddings)

        elif method_type == 'gmms':
            k_component = len(dev_cluster_idxes.keys())
            model = GMMs(n_component=k_component, seed=seed_list[i])
            pred_labels = model.predict(dev_word_embeddings)
        elif method_type == 'ac':
            k_cluster = len(dev_cluster_idxes.keys())
            model = AC(n_cluster=k_cluster)
            pred_labels = model.predict(dev_word_embeddings)

        elif method_type == 'dbscan':
            '''
            # GridSearchCV for DBSCAN factor : eps and min_sample
            sub_train_sets,sub_train_vocab = split_sub_train_set_by_dev_set(train_sets, train_vocab, dev_sets)
            sub_train_flat_word_list, sub_train_word_idxes, sub_train_cluster_idxes = get_word_idxes_and_cluster_idxes(sub_train_sets,
                                                                                                                        sub_train_vocab,
                                                                                                                        word2id)
            sub_train_word_embeddings = embedding[np.array(sub_train_word_idxes)]

            sub_train_labels = np.array([sub_train_cluster_idxes[ sub_train_vocab[word] ]for word in sub_train_flat_word_list])

            from sklearn.cluster import DBSCAN
            from sklearn import metrics
            model = DBSCAN()
            from sklearn.model_selection import StratifiedKFold, GridSearchCV
            from sklearn.metrics import make_scorer,adjusted_rand_score
            eps_list = list(floatrange(6,10,10))
            min_samples_list = list(range(1,20,10))
            param_grid = dict(eps = eps_list, min_samples = min_samples_list)
            kflod = StratifiedKFold(n_splits = 3, shuffle=True, random_state=SEED)

            def my_custom_scoring(estimator,X,y):
                y_pred = estimator.fit_predict(X)
                return  adjusted_rand_score(labels_true = y, labels_pred = y_pred)
            grid_search = GridSearchCV(model, param_grid, scoring=my_custom_scoring, n_jobs=4,cv=kflod)
            grid_result = grid_search.fit(sub_train_word_embeddings,sub_train_labels)
            print(f'Best ARI : {grid_result.best_score_}, using param : {grid_result.best_params_}')
            '''
            model = DBScan(eps=8.22, min_sample=1)
            pred_labels = model.predict(dev_word_embeddings)
        else:
            raise KeyError(f'No method type <{args.method}>')

        target_dict = {word : dev_cluster_idxes[cluster] for word, cluster in dev_vocab.items()}
        pred_dict = {word : label for word,label in zip(dev_flat_word_list,pred_labels)}
        # result_list = select_evaluate_func(['ARI','NMI','FMI'])
        _, ARI = metrics_adjusted_randn_index(pred_dict, target_dict)
        _, NMI = metrics_normalized_mutual_info_score(pred_dict,target_dict)
        _, FMI = metrics_fowlkes_mallows_score(pred_dict, target_dict)
        ari_list.append(ARI)
        nmi_list.append(NMI)
        fmi_list.append(FMI)

    ari_avg, ari_std = np.mean(ari_list), np.std(ari_list)
    nmi_avg, nmi_std = np.mean(nmi_list), np.std(nmi_list)
    fmi_avg, fmi_std = np.mean(fmi_list), np.std(fmi_list)

    log_str = '\n' + '='*40 +f'\nUse method : <{method_type}> || deal with dataset : <{datadir_name}> || embedding type: <{word_emb_select}>\n'+'='*40
    log_str += f'\nRun {run_times} times experiment...'
    log_str += f'\n[ARI] : {str(format_output(ari_list))}\n' \
               f'[FMI] : {str(format_output(fmi_list))}\n' \
               f'[NMI] : {str(format_output(nmi_list))}'
    log_str += f'\n[ARI] mean : {format_output(ari_avg)}, std : {format_output(ari_std, 2)}\n' \
               f'[FMI] mean : {format_output(fmi_avg)}, std : {format_output(fmi_std, 2)}\n' \
               f'[NMI] mean : {format_output(nmi_avg)}, std : {format_output(nmi_std, 2)}'
    log = Logger(DataConfig['log_file'], 'a')
    log.put(log_str)
    print(log_str)


