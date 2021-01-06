'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-10-26 16:36:19
 * @desc 
'''
import itertools
from typing import Any, Dict, Optional, Sequence, Tuple
from numpy.lib.scimath import sqrt
import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.special import comb


class EvalUnit(object):
    """Smalled Evaluating metrics unit
    Attribute:
        tp: True Positive item
        fp: False Positive item
        fn: False Negative item
        tn: True Negative item
        name: A label to describe specific instance
    """

    def __init__(self, tn:int=0, fp:int=0, fn:int=0, tp:int=0, name:Optional[int]=None) -> None:
        super(EvalUnit,self).__init__()
        self.name = name if name is not None else "None"
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def __repr__(self):
        desc = '\n--------- Desc EvalUnit {}---------\n'.format(self.name)
        desc += 'True Positive:{} \nTrue Negative:{} \nFalse Positive:{} \nFalse Negative:{} \n'.format(
                self.tp, self.tn, self.fp, self.fn
                )
        desc += 'Accuracy:{:.2f} \nPrecision:{:.2f} \nRecall:{:.2f} \nF1-Score:{:.2f}'.format(
                self.accuracy(), self.precision(), self.recall(), self.f1_score()
                )
        return desc

    def __add__(self,other) -> "EvalUnit":
        return EvalUnit(
            self.tn + other.tn,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tp + other.tp,
            )
        
    def __iadd__(self,other) -> "EvalUnit":
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        self.tp += other.tp
        return self

    def accuracy(self) -> float:
        return float(self.tn + self. tp) / (self.fp + self.fn + self.tn + self.tp)

    def f1_score(self) -> float:
        r = self.recall()
        p = self.precision()
        return 2 * r * p / (p + r) if p + r != 0  else 0.

    def precision(self) -> float:
        return float(self.tp) / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0.
    
    def recall(self) -> float:
        return float(self.tp) / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0.

    def metrics(self) -> Tuple[float]:
        return (self.accuracy, self.precision, self.recall, self.f1_score)


def binary_confusion_matrix_evaluate(y_true:Sequence[Any], y_pred:Sequence[Any]) -> EvalUnit:
    # import pdb; pdb.set_trace()
    tn, fp, fn, tp =  confusion_matrix(y_true,y_pred,labels = [0,1]).ravel()
    return EvalUnit(tn,fp,fn,tp)


""" ---------------- cluster Metrics ---------------- """

"""
I reimplemet adjusted_rand_index, fowlkes_mallows_scores.
In order to understand algoritmn
But In reality, We can call related API directly from sklearn.
"""
''' helper function'''
def helper_trans_to_element2clusterid(cluster:Dict) -> Dict:
    ele2cluster = {}
    for key,value in cluster.items():
        if key not in ele2cluster:
            ele2cluster[key] = []
        ele2cluster[key].append(value)
    return ele2cluster

''' '''
def cluster_confusion_matrix(pred_cluster:Dict, target_cluster:Dict) -> EvalUnit:
    """ simulate confusion matrix 
    Args:
        pred_cluster: Dict element: cluster_id （cluster_id from 0 to max_size）| predicted clusters 
        target_cluster: Dict element:cluster_id （cluster_id from 0 to max_size) | target clusters  
    Returns:
        In order to return detailed data, It will return a EvalUnit, 
    """

    
    pred_elements = list(pred_cluster.keys())
    target_elements = list(target_cluster.keys())

    it = itertools.product(pred_elements,target_elements)
    tp,fp,tn,fn = 0,0,0,0
    for x,y in it:
        if x != y:#other word
            x_cluster = pred_elements[x]
            x_cluster_ = target_elements[x]
            y_cluster = pred_elements[y]
            y_cluster_ = target_elements[y]

            if x_cluster == y_cluster and x_cluster_ == y_cluster_:
                tp += 1
            elif x_cluster != y_cluster and x_cluster_ != y_cluster_:
                tn += 1
            elif x_cluster == y_cluster and x_cluster_ != y_cluster_:
                fp += 1
            else:
                fn +=1
    return EvalUnit(tp,tn,fp,fn,'rand_index')

def get_rand_index(unit:EvalUnit) -> float:
    return unit.precision

def get_fowlkes_mallows_score(unit:EvalUnit) -> float:
    FMI = unit.tp / sqrt((unit.tp + unit.fp) * (unit.tp+ unit.fn))

def fowlkes_mallows_score(pred_cluster: Dict, target_cluster: Dict) -> float:
    unit = cluster_confusion_matrix(pred_cluster,target_cluster)
    return get_fowlkes_mallows_score(unit)


def rand_index(pred_cluster: Dict, target_cluster: Dict) -> float:
    """Use contingency_table to get RI directly
    RI = Accuracy = (TP+TN)/(TP,TN,FP,FN)
    Args:
        pred_cluster: Dict element:cluster_id （cluster_id from 0 to max_size）| predicted clusters 
        target_cluster: Dict element:cluster_id （cluster_id from 0 to max_size) | target clusters  
    Return:
        RI (float)
    """
    pred_cluster_ = helper_trans_to_element2clusterid(pred_cluster)
    target_cluster_ = helper_trans_to_element2clusterid(target_cluster)
    pred_cluster_size = len(pred_cluster_)
    target_cluster_size = len(target_cluster_)
    contingency_table = np.zeros((pred_cluster_size,target_cluster_size))
    
    for i, p_cluster in enumerate(pred_cluster_):
        for j, t_cluster in enumerate(target_cluster_):
            #find common element
            l = [*p_cluster,*t_cluster]
            contingency_table[i][j] = len(l) - len(set(l))
    s = comb(np.sum(contingency_table), 2)
    a = 0
    for i in np.nditer(contingency_table):
        a += comb(i,2)
    return a/s

def adjusted_rand_index(pred_cluster:Dict, target_cluster:Dict):
    """Docstring
    Using Contingency Matrix to calculate ARI
    Continggency Matrix
    --------------------------------
    XY  | Y_1  Y_2  ...  Y_s  | sums
    --------------------------------
    X_1 | n_11 n_12 ...  n_1s | a_1
    X_2 | n_21 n_22 ...  n_2s | a_2
    ... | ...  ...  ...  ...  | ...
    X_r | n_r1 n_r2 ...  n_rs | a_r
    sum | b_1  b_2  ...  b_s  |
    --------------------------------
    f(x) = comb(x,2)
    ARI = [ sum f(n_ij) - sum f(a_ij) * sum f(b_ij) / f(n) ] /
            [0.5 * [ sum f(a_ij) + sum f(b_ij)] - sum f(a_ij) * sum f(b_ij) / f(n)]
    
    Args:
        pred_cluster: Dict cluster_id: List[element] （cluster_id from 0 to max_size）| predicted clusters 
        target_cluster: Dict cluster_id: List[element] （cluster_id from 0 to max_size) | target clusters  
    Return:
        ARI (float)
    """
    
    pred_cluster_ = helper_trans_to_element2clusterid(pred_cluster)
    target_cluster_ = helper_trans_to_element2clusterid(target_cluster)
    pred_cluster_size = len(pred_cluster_)
    target_cluster_size = len(target_cluster_)
    contingency_table = np.zeros((pred_cluster_size, target_cluster_size))

    for i, p_cluster in enumerate(pred_cluster_):
        for j, t_cluster in enumerate(target_cluster_):
            #find common element
            l = [*p_cluster,*t_cluster]
            contingency_table[i][j] = len(l) - len(set(l))

    s = comb(np.sum(contingency_table), 2)
    ij = 0
    for i in np.npiter(contingency_table):
        ij += comb(i,2)
    pred_sum = np.sum(contingency_table, axis=1)
    target_sum = np.sum(contingency_table, aixs=0)
    
    pred_comb_sum = 0
    for i in np.npiter(pred_sum):
        pred_comb_sum += comb(i,2)
    target_comb_sum = 0
    for i in np.npiter(target_sum):
        target_comb_sum += comb(i,2)
    tmp = pred_comb_sum * target_comb_sum / s
    ARI = (ij - tmp) / (0.5*(pred_comb_sum+target_comb_sum) - tmp)

    return ARI



"""
Below function is inference of sklearn
I changed the input data type slightly
"""


'''helper function'''
def helper_trans_to_labelsequence(cluster:Dict,cluster_:Dict)-> Any:
    keys = cluster.keys()
    label_sequence = []
    label_sequence_ = []
    for element in keys:
        label_sequence.append(cluster[element])
        label_sequence_.append(cluster_[element])
    return np.array(label_sequence), np.array(label_sequence_)

def metrics_adjusted_randn_index(pred_cluster:Dict, target_cluster:Dict) -> Any:
    pred_sequence,target_sequence = helper_trans_to_labelsequence(pred_cluster,target_cluster)
    return 'ARI', sklearn.metrics.adjusted_rand_score(labels_true = target_sequence, labels_pred = pred_sequence)

def metrics_normalized_mutual_info_score(pred_cluster:Dict, target_cluster:Dict) -> Any:
    pred_sequence,target_sequence = helper_trans_to_labelsequence(pred_cluster,target_cluster)
    return 'NMI', sklearn.metrics.normalized_mutual_info_score(labels_true = target_sequence, labels_pred = pred_sequence)

def metrics_fowlkes_mallows_score(pred_cluster:Dict, target_cluster:Dict) ->Any:
    pred_sequence,target_sequence = helper_trans_to_labelsequence(pred_cluster,target_cluster)
    return 'FMI', sklearn.metrics.fowlkes_mallows_score(labels_true = target_sequence, labels_pred = pred_sequence)

#through config to add function list
def select_evaluate_func(select:Sequence[str]) -> Any:
    func_list = []
    for i in select:
        if i == 'ARI':
            func_list.append(metrics_adjusted_randn_index)
        elif i == 'NMI':
            func_list.append(metrics_normalized_mutual_info_score)
        elif i == 'FMI':
            func_list.append(metrics_fowlkes_mallows_score)
        else:
            raise KeyError
    return func_list

if __name__ == '__main__':
    print(1)