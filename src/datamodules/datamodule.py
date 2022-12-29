from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets, Dataset

import torch
import pandas as pd
from math import floor
import random
from numpy.random import choice
import copy

import json
with open('../../../../data/features.txt') as f:
    FEATURES = json.loads(f.read())
    
MAX_LENGTH = 250
    
    
def get_length(dataset):
    length = 0
    while True:
        try:
            next(dataset)
            length += 1
        except:
            return length


def collate_fn(data):
    collated_data = {}
    for key in data[0].keys():
        if key in ['input_ids','attention_mask']:
            max_length = max([len(item[key]) for item in data])
            collated_data[key] = []
            for item in data:
                sentence = [0]*max_length
                sentence[:len(item[key])] = item[key]
                collated_data[key].append(sentence)
        else:
            collated_data[key] = [item[key] for item in data]
    return collated_data

    
class LanguageDataset(Dataset):
    
    def __init__(self, data_dir, split, feature_value, train_samples, dense_features, insert_start_tok):

        self.insert_start_tok = insert_start_tok
        with open(data_dir+'data_info.json') as f:
            feature_values = json.loads(f.read()).keys()

        self.dataset = data_dir.split('/')[-2]
        self.datasets = []
        self.lengths = {'start':[],'current':[]}
        
        def _load_dataset(data_file, split, train_samples):
            dataset = load_dataset('json', data_files=data_file, streaming=True)['train']
            length = sum(1 for e in open(data_file,'r'))
            if train_samples != None:
                assert length>train_samples
                length = train_samples/0.9
            train_length = floor(length*0.9)
            val_length = max(1,floor(length*0.05))
            test_length = max(1,floor(length*0.05))
            if split == 'train':
                return dataset.take(train_length), train_length
            elif split == 'val':
                return dataset.skip(train_length).take(val_length), val_length
            elif split == 'test':
                return dataset.skip(train_length+val_length).take(test_length), test_length

        if feature_value == None or split == 'test':
            
            for feature_value in list(feature_values):
                if dense_features:
                    dataset, length = _load_dataset(data_dir+'files/'+feature_value+'_data_filtered.json', split, train_samples)
                else:
                    dataset, length = _load_dataset(data_dir+'files/'+feature_value+'_data.json', split, train_samples)
                self.datasets.append(dataset)
                self.lengths['start'].append(length)
                self.lengths['current'].append(length)
        else:
            if dense_features:
                dataset, length = _load_dataset(data_dir+'files/'+feature_value+'_data_filtered.json', split, train_samples)
            else:
                dataset, length = _load_dataset(data_dir+'files/'+feature_value+'_data.json', split, train_samples)
            self.datasets.append(dataset)
            self.lengths['start'].append(length)
            self.lengths['current'].append(length)
                
    def __getitem__(self, idx):
        if idx == 0:
            self.iter_datasets = []
            for dataset in self.datasets:
                self.iter_datasets.append(iter(dataset))
            self.lengths['current'] = copy.deepcopy(self.lengths['start'])
            
        dataset_idx = choice(list(range(len(self.datasets))),
                             p=[length/sum(self.lengths['current']) for 
                                length in self.lengths['current']])
            
        raw_item = next(self.iter_datasets[dataset_idx])
        self.lengths['current'][dataset_idx] -= 1
        
        item = {}
        for key in raw_item.keys():
            if key[-3:]=='_id':
                if key == 'genre_id':
                    raw_item[key] = [i for i in raw_item[key] if i in [1, 3, 6, 7, 9, 14, 17, 18, 19, 22]]
                    item[key] = random.choice(raw_item[key])
                else:
                    item[key] = raw_item[key]
            elif key == 'sentence':
                if self.insert_start_tok:
                    item['input_ids'] = [50256]+raw_item[key][:MAX_LENGTH-1] # 50256 == BOS id
                else:
                    item['input_ids'] = raw_item[key][:MAX_LENGTH]
                item['attention_mask'] = [1]*len(item['input_ids'])
        return item
        
    def __len__(self):
        return sum(self.lengths['start'])
    

class DataModule(LightningDataModule):
    
    def __init__(
        self,
        dataset: str = 'movie_dialogue',
        feature_value: str = None,
        train_samples: int = None, # per feature value
        dense_features: bool = False,
        insert_start_tok: bool = False,
        data_dir: str = "data/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data = []
            for split in ['train','val','test']:
                data.append(LanguageDataset(self.hparams.data_dir+'/'+self.hparams.dataset+'/',
                                            split, self.hparams.feature_value, 
                                            self.hparams.train_samples, self.hparams.dense_features,
                                            self.hparams.insert_start_tok))
            self.data_train, self.data_val, self.data_test = data
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            collate_fn=collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            collate_fn=collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            collate_fn=collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
