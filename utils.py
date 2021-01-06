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

if __name__ == '__main__':
    pass