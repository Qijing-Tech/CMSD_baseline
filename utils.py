# -*- coding: utf-8 -*-
"""
@Time ：　2020/12/31
@Auth ：　xiaer.wang
@File ：　utils.py
@IDE 　：　PyCharm
"""
from typing import List,Dict,Tuple
from collections import Counter
import random
import torch
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
import networkx as nx

def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    """
    Args:
        seed: Random Seed
        deterministic:
                Deterministic operation may have a negative single-run performance impact, depending on the composition of your model.
                Due to different underlying operations, which may be slower, the processing speed (e.g. the number of batches trained per second) may be lower than when the model functions nondeterministically.
                However, even though single-run speed may be slower, depending on your application determinism may save time by facilitating experimentation, debugging, and regression testing.
        benchmark: whether cudnn to find most efficient method to process data
                If no difference of the size or dimension in the input data, Setting it true is a good way to speed up
                However, if not,  every iteration, cudnn has to find best algorithm, it cost a lot
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def split_sub_train_set_by_dev_set(train_sets : List[List[str]], train_vocab : Dict,
                                   dev_sets : List[List[str]], is_print=True) -> Tuple[List[List[str]], Dict]:
    '''
        按 dev_sets 每个类别的词频，划分 train_sets ,
        使得切出一个与　验证集　聚类个数相同总词数相近的　子词集
    '''
    len_cnt_dict = Counter([len(x) for x in dev_sets])

    reconstruct_train_sets = {}
    for word_list in train_sets:
        if len(word_list) not in reconstruct_train_sets:
            reconstruct_train_sets[len(word_list)] = [word_list]
        else:
            reconstruct_train_sets[len(word_list)].append(word_list)

    leave_word_len = set(reconstruct_train_sets.keys()) - set(len_cnt_dict.keys())

    sub_train_word_lists = []
    sub_train_vocab = {}
    for word_len, len_cnt in len_cnt_dict.items():
        if word_len in reconstruct_train_sets:
            select_list = random.sample(reconstruct_train_sets[word_len], len_cnt)
            sub_train_word_lists.extend(select_list)
            sub_train_vocab.update({word : train_vocab[word] for word_list in select_list for word in word_list})
        else: #have no len -> select from leave_word_len
            avaliable_select_list = []
            for _len in leave_word_len:
                #用长度不超过２的替代
                if abs(_len - len_cnt) <= 2 and \
                        len(reconstruct_train_sets[_len]) >= len_cnt:
                    avaliable_select_list = reconstruct_train_sets[_len]
                    leave_word_len.remove(_len)
                    break
            if avaliable_select_list != []:
                select_list = random.sample(avaliable_select_list, len_cnt)
                sub_train_word_lists.extend(select_list)
                sub_train_vocab.update({word: train_vocab[word] for word_list in select_list for word in word_list})
            else:
                print(f'Give up...');pass

    if is_print:
        total_vocab_nums = len([word for word_list in dev_sets for word in word_list])
        sub_vocab_nums = len([word for word_list in sub_train_word_lists for word in word_list])
        print(f'Target cluster nums : {len(dev_sets)}, vocab nums : {total_vocab_nums}')
        print(f'Obtained cluster nums : {len(sub_train_word_lists)}, vocab nums : {sub_vocab_nums}')

    return sub_train_word_lists, sub_train_vocab

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

def grid_search_model_params(word_list : List[List], word_vocab : Dict, word2id : Dict, embedding : np.ndarray,
                             model, learn_params : Dict, score_index = 'ARI', seed = 1234) -> Dict:

    sub_train_flat_word_list, sub_train_word_idxes, sub_train_cluster_idxes = get_word_idxes_and_cluster_idxes(word_list,
                                                                                                                word_vocab,
                                                                                                                word2id)
    sub_train_word_embeddings = embedding[np.array(sub_train_word_idxes)]
    sub_train_labels = np.array([sub_train_cluster_idxes[ word_vocab[word] ]for word in sub_train_flat_word_list])

    kflod = StratifiedKFold(n_splits = 3, shuffle=True, random_state=seed)

    def my_custom_scoring(estimator,X,y):
        y_pred = estimator.fit_predict(X)
        if score_index == 'ARI':
            return  metrics.adjusted_rand_score(labels_true = y, labels_pred = y_pred)
        elif score_index == 'FMI':
            return metrics.fowlkes_mallows_score(labels_true= y, labels_pred= y_pred)
        elif score_index == 'NMI':
            return metrics.normalized_mutual_info_score(labels_true= y, labels_pred= y_pred)
        else:
            raise NotImplementedError

    grid_search = GridSearchCV(model, learn_params, scoring=my_custom_scoring, n_jobs=4,cv=kflod)
    grid_result = grid_search.fit(sub_train_word_embeddings,sub_train_labels)
    print(f'Best {score_index} : {grid_result.best_score_}, using param : {grid_result.best_params_}')

    return grid_result.best_params_

def construct_graph_by_words(word_lists : List, word_embeddings : np.ndarray):

    #有个问题 ： weight 用什么衡量最好 ？
    def get_distance(v1, v2, measure='euclidean'):
        if measure == 'euclidean':
            return np.linalg.norm(v1 - v2)
        elif measure == 'cosine':
            return np.dot(v1, v2) / np.linalg.norm(v1) * np.linalg.norm(v2)
        elif measure == 'dot':
            return np.dot(v1, v2)
        else:
            raise NotImplementedError

    g = nx.Graph()
    for i, word_prev in enumerate(word_lists):
        for j, word_after in enumerate(word_lists):
            if i < j:
                p = random.random()
                if p >= 0.:
                    g.add_edge(word_prev, word_after, weight= get_distance(word_embeddings[i], word_embeddings[j], measure='euclidean'))

    return g

if __name__ == '__main__':
    pass