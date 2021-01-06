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