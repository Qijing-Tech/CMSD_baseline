# -*- coding: utf-8 -*-
"""
@Time ：　2021/12/31
@Auth ：　xiaer.wang
@File ：　config.py
@IDE 　：　PyCharm
"""
# unspervised model config
'''
data_name : [sm_HIT_syn, ex_HIT_syn]
word_emb_select : [combined.embed, Tencent_combined.embed]
method_type : [kmeans, gmms, dbscan, ac，optics]
'''

from pathlib import Path
cwd = Path.cwd()

# ------------- path config ------------- #
DATA_ROOT = cwd.joinpath('data')

# ------------- data config ------------- #
DataConfig = {
    'data_name' : 'sm_HIT_syn',
    'word_emb_select' :'combined.embed',
    'method_type' : 'dbscan',
    'run_times' : 5,
    'seed_list': [1,1234,1314,2020,2021],
    'log_file' : 'record_v2.txt',
}