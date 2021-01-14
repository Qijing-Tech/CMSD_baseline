'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-01-06 18:05:44
 * @desc termnial args
'''


import argparse

parser = argparse.ArgumentParser(description="Process some Command")
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--embed', type=str, default='combined.embed')

#set for supervised method, such as Learning to Cluster(L2C)
parser.add_argument('--gpuid', nargs="+", type=int, default=[-1],
                    help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
parser.add_argument('--model_type', type=str, default='l2c', help="l2c(default) ...")
parser.add_argument('--model_name', type=str, default='L2C_v0',
                    help="L2C_v0(default)|L2C_v1|L2C_v2|L2C_v4|L2C_v8 ...")
parser.add_argument('--loss_type', type=str, default='MCL', choices=['KCL', 'MCL', 'DPS'],
                        help="KCL|MCL(default)|DPS(Dense-Pair Similarity)")
parser.add_argument('--saveid', type=str, default='', help="The appendix to the saved model")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--schedule', nargs="+", type=int, default=[5, 10],
                    help="The list of epoch numbers to reduce learning rate by factor of 0.1")
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
parser.add_argument('--epochs', type=int, default=15, help="End epoch")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--skip_eval', dest='skip_eval', default=False, action='store_true',
                    help="Only do the evaluation after training is done")
parser.add_argument('--out_dim', type=int, default=-1,
                    help="Output dimension of network. Default:-1 (Use ground-truth)")
parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")

#for SPN
parser.add_argument('--use_SPN', dest='use_SPN', default=False, action='store_true',
                    help="Use Similarity Prediction Network")
parser.add_argument('--SPN_model_type', type=str, default='l2c', help="This option is only valid when use_SPN=True")
parser.add_argument('--SPN_model_name', type=str, default='L2C_v0', help="This option is only valid when use_SPN=True")
parser.add_argument('--model_save_path', type=str, default='ckpt', help='Save DSP model')
parser.add_argument('--SPN_pretrained_model', type=str, default='l2c_L2C_v0.model.pth',
                    help="This option is only valid when use_SPN=True")
