# -*- coding: utf-8 -*-
"""
@Time ：　2020/12/31
@Auth ：　xiaer.wang
@File ：　run_test.py
@IDE 　：　PyCharm
"""
from pathlib import Path
from dataloader import DataSetDir
from method import Kmeans,GMMs,DBScan,AC,Optics
from sklearn.cluster import OPTICS,DBSCAN
import numpy as np
from evaluate import select_evaluate_func,metrics_adjusted_randn_index,\
    metrics_normalized_mutual_info_score,metrics_fowlkes_mallows_score
from config import DataConfig,DATA_ROOT
from utils import set_random_seed
from logger import Logger
from utils import split_sub_train_set_by_dev_set, get_word_idxes_and_cluster_idxes, \
    grid_search_model_params, construct_graph_by_words
from args import parser
args = parser.parse_args()
cwd = Path.cwd()
SEED = 2020

def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]
def format_output(data, bit = 2):

    if isinstance(data, float):
        return round(data * 100, bit)
    elif isinstance(data, list):
       return [round(d * 100, bit)  for d in data]

if __name__ == '__main__':

    DataConfig['data_name'] = args.data
    DataConfig['method_type'] = args.method
    DataConfig['word_emb_select'] = args.embed
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

    run_times = DataConfig['run_times']
    seed_list = DataConfig['seed_list']
    ari_list, nmi_list, fmi_list = [], [], []
    for i in range(run_times):
        # TODO : model
        set_random_seed(seed=seed_list[i])
        dev_flat_word_list, dev_word_idxes, dev_cluster_idxes = get_word_idxes_and_cluster_idxes(dev_sets, dev_vocab,word2id)
        dev_word_embeddings = embedding[np.array(dev_word_idxes)]

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
        elif method_type == 'optics':
            if i == 0:
                sub_train_sets, sub_train_vocab = split_sub_train_set_by_dev_set(train_sets, train_vocab, dev_sets)
                min_samples_list = list(range(1, 10))
                learn_params = dict(min_samples=min_samples_list)
                model = OPTICS()
                best_params = grid_search_model_params(sub_train_sets, sub_train_vocab, word2id,
                                                       embedding,model,learn_params,score_index='ARI')
                best_min_sample = best_params['min_samples']
                del sub_train_sets, sub_train_vocab

            model = Optics(min_sample = best_min_sample)
            pred_labels = model.predict(dev_word_embeddings)

        elif method_type == 'dbscan':

            if i == 0:
                sub_train_sets, sub_train_vocab = split_sub_train_set_by_dev_set(train_sets, train_vocab, dev_sets)
                eps_list = list(floatrange(0.1, 10, 10))
                min_samples_list = list(range(1, 10, 10))
                learn_params = dict(eps = eps_list, min_samples = min_samples_list)
                model = DBSCAN()
                best_params = grid_search_model_params(sub_train_sets, sub_train_vocab, word2id,
                                                       embedding,model,learn_params,score_index='ARI')
                best_min_sample = best_params['min_samples']
                best_eps = best_params['eps']
                del sub_train_sets, sub_train_vocab

            model = DBScan(eps = best_eps, min_sample = best_min_sample)
            pred_labels = model.predict(dev_word_embeddings)

        elif method_type == 'louvain':
            import community
            graph = construct_graph_by_words(dev_flat_word_list, dev_word_embeddings)
            map_idx = {i: word for i,word in enumerate(dev_flat_word_list)}
            partition = community.best_partition(graph)

            pred_labels = [partition[word] for idx, word in map_idx.items()]

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
    log_str += f'\n[ARI] mean : {format_output(ari_avg)}, std : {format_output(ari_std)}\n' \
               f'[FMI] mean : {format_output(fmi_avg)}, std : {format_output(fmi_std)}\n' \
               f'[NMI] mean : {format_output(nmi_avg)}, std : {format_output(nmi_std)}'

    log = Logger(DataConfig['log_file'], 'a')
    log.put(log_str)
    print(log_str)


