# -*- coding: utf-8 -*-
"""
@Time ：　2021/1/12
@Auth ：　xiaer.wang
@File ：　run_l2c.py
@IDE 　：　PyCharm
"""

from method import L2C
from args import parser
import os
from dataloader import DataSetDir, EmbeddingDataSet
from config import DataConfig,DATA_ROOT
from torch.utils.data import DataLoader
from utils import split_sub_train_set_by_dev_set
from run import format_output
from logger import Logger
args = parser.parse_args()

def get_data_loader(args, datadir):

    word2id = datadir.word2id
    embedding = datadir.embedding_vec

    in_dim = embedding.shape[1]
    dev_vocab = datadir.dev_dataset.vocab
    dev_sets = datadir.dev_dataset.raw_sets

    train_vocab = datadir.train_dataset.vocab
    train_sets = datadir.train_dataset.raw_sets

    #split dev dataset
    sub_dev_sets, sub_dev_vocab, train_sets, train_vocab = split_sub_train_set_by_dev_set(train_sets, train_vocab, train_sets, sample_ratio=0.3)


    # For train SPN
    train_dataset = EmbeddingDataSet(train_sets, train_vocab, word2id, embedding)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    dev_dataset = EmbeddingDataSet(sub_dev_sets, sub_dev_vocab, word2id, embedding)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    test_dataset = EmbeddingDataSet(dev_sets, dev_vocab, word2id, embedding)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,shuffle=False)

    return in_dim, train_loader, dev_loader, test_loader, train_dataset, dev_dataset, test_dataset


def run_model(args):

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    model_path = os.path.join(args.model_save_path, "%s#%s#%s#DPS.model.pth" % (args.data,args.embed, args.model_name))
    if not os.path.exists(model_path):
        args.is_train = True
    # DATA
    DataConfig['data_name'] = args.data
    DataConfig['method_type'] = args.method
    DataConfig['word_emb_select'] = args.embed
    datadir_name = DataConfig['data_name']
    data = DATA_ROOT.joinpath(datadir_name)
    word_emb_select = DataConfig['word_emb_select']
    method_type = DataConfig['method_type']
    datadir = DataSetDir(data, word_emb_select=word_emb_select)
    in_dim, train_loader, eval_loader, test_loader ,train_dataset, eval_dataset, test_dataset = get_data_loader(args,  datadir)
    #MODEL
    origin_loss_type = args.loss_type
    origin_out_dim = args.out_dim
    if args.is_train:
        args.out_dim = 2
        args.loss_type = 'DPS'
        args.saveid = args.loss_type if args.saveid == '' else args.saveid
        tgt_class = train_dataset.num_class
        print(f'STEP1: Train SPN on <{args.data}> dataset with <{args.embed}>')
        model = L2C(args, in_dim = in_dim, out_dim = args.out_dim)
        model.train(args, train_loader, eval_loader,tgt_class )


    if origin_out_dim < 0:  # Use ground-truth number of classes/clusters
        args.out_dim = train_dataset.num_class

    args.loss_type = origin_loss_type
    args.use_SPN = True
    args.skip_eval = True
    model = L2C(args, in_dim=in_dim, out_dim=args.out_dim, SPN_model=model_path)
    args.prin_freq = 0
    model.train(args, train_loader, eval_loader, train_dataset.num_class)
    cluster_info = model.predict(test_loader, args, test_dataset.num_class)
    log_str = '\n' + '=' * 40 + f'\nUse method : <{method_type}> || deal with dataset : <{datadir_name}> || embedding type: <{word_emb_select}>\n' + '=' * 40
    log_str += f"\n[ARI] : {format_output(cluster_info['ARI'])}\n" \
               f"[FMI] : {format_output(cluster_info['AMI'])}\n" \
               f"[NMI] : {format_output(cluster_info['NMI'])}"

    log = Logger(DataConfig['log_file'], 'a')
    log.put(log_str)
    print(log_str)


def init_set(args):
    args.use_gpu = args.gpuid[0] >= 0
    args.start_epoch = 0
    args.cluster2Class = None
    args.SPN = None
    args.is_train = False
    return args

if __name__ == '__main__':
    args = init_set(args)
    run_model(args)