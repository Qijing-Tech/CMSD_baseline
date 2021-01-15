
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-10 10:41:30
@desc 
    : Generate DataSet and DataLoader for Classifier
'''

from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple, Any
from copy import copy
import numpy as np
import pickle
import config
from collections import OrderedDict
from torch.utils.data import Dataset
from utils import get_word_idxes_and_cluster_idxes
pattern = "(?<=\')[^\|\']*\|\|[^\|\']*?(?=\')"


class DataSetDir(object):
    """ class: DataSet Dir including training data and dev data """
    def __init__(self,dir_file_path:Path,dir_structure:Optional[Dict] = None, word_emb_select:Optional[str] = None):
        self.path = dir_file_path
        self.name = dir_file_path.name
        self.train_file_name = 'train.set'
        self.test_file_name = 'test.set'
        self.dev_file_name = 'dev.set'
        if dir_structure:
            self.train_file_name = dir_structure['train']
            self.test_file_name = dir_structure['test']
            self.dev_file_name = dir_structure['dev']
        
        word_emb_file = None
        if word_emb_select == None or word_emb_select == 'combined.embed':
            word_emb_file = dir_file_path.joinpath('combined.embed')
        elif word_emb_select == 'fastText-subword':
            word_emb_file = dir_file_path.joinpath('combined.fastText-with-subword.embed')
        elif word_emb_select == 'fastText-no-subword':
            word_emb_file = dir_file_path.joinpath('combined.fastText-no-subword.embed')
        elif word_emb_select == 'Tencent_combined.embed':
            word_emb_file = dir_file_path.joinpath('Tencent_combined.embed')
        self.word2id, self.embedding_vec = self._read_embed_info(str(word_emb_file))
        
        self.train_dataset = DataSet(
                                dir_file_path.joinpath(self.train_file_name),
                                self.name+"_training"
                            )
        self.test_dataset = DataSet(
                                dir_file_path.joinpath(self.test_file_name),
                                self.name+"_testing"
                            )
        self.dev_dataset = DataSet(
                                dir_file_path.joinpath(self.dev_file_name),
                                self.name+"_dev"
                            )
            
    def _read_embed_info(self,filepath:str):
        """Read word embeding file"""
        with open(filepath, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
        line_zero = lines[0]
        vocab_size, dim_size = [int(i) for i in line_zero.strip().split(' ')]
        word2id = {}
        word2id['PAD'] = 0
        word2id['UNK'] = 1
        embed_matrix = [[0]*dim_size,[0]*dim_size]
        for idx,line in enumerate(lines[1:]):
            t = line.strip()
#             print(t)
            word, t = t.split('||')
#             print(word)
#             print(t)
            t = t.split(' ')
            word2id[word] = 2+idx
            nums = [ eval(i) for i in t[1:]]
            embed_matrix.append(nums)
        
        embed_np_matrix = np.array(embed_matrix,dtype=np.float32)
        return word2id, embed_np_matrix 
    
class DataSet(object):
    """class: description of Raw Data"""
    def __init__(self,file_path:Path,name:str):
        self.name = name
        self.vocab,self.raw_sets,self.max_set_size,self.min_set_size,self.average_set_size = self._initilize(file_path=str(file_path))
      
    def __iter__(self):
        for word_set in self.raw_sets:
            yield word_set
    
    def __repr__(self):
        s="-------------Description of Dataset-------------\n"
        return s+'Raw DataSet Name {} \n vocab size {} \n num of sets {} \n the size of biggest set {} \n the size of smallest set {} \n the average size of all sets {}'.format(\
                self.name,len(self.vocab),len(self.raw_sets),self.max_set_size,self.min_set_size,self.average_set_size)
  
    def _initilize(self,file_path:str)->None:
        '''Initialize Dataset from raw string
            @Param file_path:Raw Data file path
            @Return
                vocab: vocabulary for all word in dataset 
                type: dict {'word':'cluster_id'}

                allsets: Graund Trueth clusters
                type: list [ ["U.S.A","U.S"] ,[]]

                max_set_size: maximun of set size in allsets
                type: int

                min_set_size: minimun of set size in allsets
                type: int

                average_set_size: average of set size in allsets
                type: float
        '''
        vocab = {}
        allsets = []
        max_set_size = -1
        min_set_size = 1e10
        sum_set_size = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            pos = line.find(' ')
            words = eval(line[pos+1:])
            size = len(words)
            item = []
            for i in words:
                word, cluster= i.strip().split('||')
                vocab[word] = cluster
                item.append(word)
            max_set_size = max_set_size if max_set_size>size else size
            min_set_size = min_set_size if min_set_size<size else size
            sum_set_size += size
            allsets.append(item)

        return vocab,allsets,max_set_size,min_set_size,sum_set_size/len(allsets)
    

class DataItemSampler(object):
    """Interface of various sampler"""
    def sample(self,wordpool:Dict, wordset:List[str], negative_sample_size:int)->Tuple[List[Tuple],int]:
        raise NotImplementedError
    
    def _negative_sample(self,wordpool: Dict, wordset: List[str], negative_sample_size:int)-> List[str]:
        pos_cluster = wordpool[wordset[0]]
        sample_word_pool = [ word for word in wordpool.keys() if wordpool[word] != pos_cluster]
        #sample size bigger than sample pool
        if negative_sample_size > len(sample_word_pool):
            return sample_word_pool
        else:
            return random.sample(sample_word_pool,negative_sample_size)


class Sample_size_repeat_size(DataItemSampler):
    """A sample method to sample dataitem : For one original word set, randomly get subset size, and repeat this size. 
        This is the strategy to original AAAI submission."""
    def __init__(self) -> None:
        super(Sample_size_repeat_size,self).__init__()

    def sample(self,wordpool:Dict, wordset:List[str], negtive_sample_size:int)->Tuple[List[Tuple],int]:
        ans = []
        setsize = len(wordset)

        if setsize == 1:
            # for only one word set, we generate one positive item and one negative item
            pos_word = wordset[0]
            neg_word = random.choice(list(wordpool.keys()))
            while neg_word == pos_word:
                neg_word = random.choice(wordpool.keys())
            ans = [
                (pos_word,pos_word,1),
                (pos_word,neg_word,0)
            ]
            return ans,1,1,1

        # random choice subsize
        new_set =  wordset.copy()
        subset_size = random.choice(range(1,setsize))
        
        pos_word_set = new_set[:subset_size]
        pos_word = new_set[subset_size]

        ans = [(pos_word_set, pos_word, 1)]

        neg_word_list = self._negative_sample(wordpool,wordset,negtive_sample_size)
        for neg_word in neg_word_list:
            ans.append((pos_word_set,neg_word,0))
        pos_item_nums = 1
        neg_item_nums = pos_item_nums * negtive_sample_size
        return ans,subset_size, pos_item_nums, neg_item_nums

class Sample_vary_size_enumerate(DataItemSampler):
    """A sample method to sample dataitem : For one original word set, enunmerate all subset size and get item
    """
    def __init__(self) -> None:
        super(Sample_vary_size_enumerate,self).__init__()

    def sample(self, wordpool: Dict, wordset: List[str], negative_sample_size:int) -> Tuple[List[Tuple],int]:
        """c"""
        pass


def select_sampler(name:str)-> DataItemSampler:
    if name == "sample_vary_size_enumerate":
        return Sample_vary_size_enumerate()
    elif name == "sample_size_repeat_size":
        return Sample_size_repeat_size()
    else:
        raise KeyError

class DataItemSet(object):
    """DataItemSet  Generate Training and Testing data item"""
    def __init__(self, dataset:DataSet, sampler:DataItemSampler, negative_sample_size:int = 10)->None:
        self.negative_sample_size = negative_sample_size
        self.vocab = dataset.vocab
        self.sampler = sampler
        self.average_set_size = -1
        self.max_set_size = -1
        self.min_set_size = -1
        self.dataitems = self._initialize(dataset)

        '''
        item in dataitems is a tuple
        (
            'Set': [ 'word1' , 'word3' ,..., 'wordn']
            'word' : word (waiting to classified)
            'label' : 1 or 0 (positive item or negative item)
        )
        '''

    def _initialize(self,dataset:DataSet)->None:
        dataitem = []
        subset_size_list = []
        neg_item_num = 0
        pos_item_num = 0

        for wordset in dataset:
            subitems, subset_size, pos_item_size, neg_item_size = self.sampler.sample(self.vocab,wordset,self.negative_sample_size)

            neg_item_num += neg_item_size
            pos_item_num += pos_item_size

            dataitem.extend(subitems)
            subset_size_list.append(subset_size)
        
        self.pos_item_num = pos_item_num
        self.neg_item_num = neg_item_num
        self.average_set_size = 1.0 * sum(subset_size_list)/len(subset_size_list)
        self.max_set_size = max(subset_size_list)
        self.min_set_size = min(subset_size_list)
        return dataitem

    def __repr__(self)->None:
        s="-----------Description of DataItemSet--------------\n"
        return s+'nums of dataitem {} \n the size of biggest set item {} \n the size of smallest set item {} \n the average size of all sets item {:.2f} \n negtive items in dataset:{} \n positive items in dataset:{}'.format(\
                len(self.dataitems),self.max_set_size,self.min_set_size,self.average_set_size,self.neg_item_num, self.pos_item_num)

    def __len__(self) -> int:
        return len(self.dataitems)
    
    def __iter__(self)->None:
        for i in self.dataitems:
            yield i
        
    def __getitem__(self,index):
        return self.dataitems[index]

    @classmethod
    def save(cls,path:Path):
        """save dataitem"""
        with open(path, 'wb', encoding='utf-8') as f:
            pickle.dump(cls,f)
        # pass

    @classmethod
    def load(cls,path:Path):
        """load dataitem"""
        with open(path, 'rb', encoding='utf-8') as f:
            ans = pickle.load(f)
        return ans


class Dataloader(object):
    """Dataloader to get batch size dataitem"""
    def __init__(self, dataitems:DataItemSet, word2id, batch_size:int) -> None:
        self.data = dataitems
        self.l = len(self.data)//batch_size if len(self.data)%batch_size == 0 else len(self.data)//batch_size + 1
        self.batch_size = batch_size
        self.word2id = word2id

    def __len__(self):
        return self.l

    def __iter__(self):
        ixs = list(range(0,len(self.data)))
        random.shuffle(ixs)
        batch_old_word_set, batch_new_word_set, batch_label = [], [], []
        for index,ix in enumerate(ixs):
            word_set, waiting_word, label = self.data[ix]
            word_id_set = [ self.word2id[word] for word in word_set]
            waiting_word_id = self.word2id[waiting_word]
            new_word_id_set = word_id_set.copy()
            new_word_id_set.append(waiting_word_id)
            
            batch_old_word_set.append(word_id_set)
            batch_new_word_set.append(new_word_id_set)
            batch_label.append(label)

            if (index+1) % self.batch_size == 0 or index == len(self.data) - 1:
                (batch_old_word_set_, old_mask), (batch_new_word_set_,new_mask) = set_padding(batch_old_word_set), set_padding(batch_new_word_set)
                # batch_label = self._trans_label(batch_label)
                yield batch_old_word_set_, old_mask, batch_new_word_set_, new_mask, batch_label
                batch_old_word_set, batch_new_word_set, batch_label = [],[],[]
    
    # Error method
    def split(self, q:float=0.2) -> Tuple[Any]:
        dataitem = copy(self.data)
        ixs = list(range(0,len(self.data)))
        random.shuffle(ixs)
        sub_l = int(self.l * q)
        return [
                Dataloader(dataitem[:sub_l], self.word2id, self.batch_size),
                Dataloader(dataitem[sub_l:], self.word2id, self.batch_size)
            ]



class EmbeddingDataSet(Dataset):

    def __init__(self, word_list, vocab_dict, word2id, embedding):
        flat_word_list, word_idxes, cluster_idxes = get_word_idxes_and_cluster_idxes(word_list, vocab_dict, word2id)
        self.word_labels = [cluster_idxes[ vocab_dict[word] ] for word in flat_word_list]
        self.word_embeddings = embedding[np.array(word_idxes)]
        self.num_class = len(cluster_idxes.keys())
        assert self.word_embeddings.shape[0] == len(self.word_labels)

    def __getitem__(self, idx):

        return self.word_embeddings[idx], self.word_labels[idx]

    def __len__(self):
        return len(self.word_labels)


    


def test_CSKB():
    datasetdir = DataSetDir(config.CSKB_DIR_PATH)
    train_datasetitem = DataItemSet(
                    dataset=datasetdir.train_dataset,
                    sampler = select_sampler(config.dataconfig['sample_strategy']),
                    negative_sample_size = config.dataconfig['negative_sample_size']
                )
    train_dataloader = Dataloader(
                    dataitems=train_datasetitem, 
                    word2id=datasetdir.word2id,
                    batch_size=config.trainingconfig['batch_size']
                )

if __name__ == '__main__':
    import pdb;pdb.set_trace()
    test_CSKB()
    # test_dataitemset()
    # test_dataloader()

