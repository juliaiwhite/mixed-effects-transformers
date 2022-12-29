from typing import List, Optional, Tuple
import torch
import pandas as pd
from math import floor
import random
from numpy.random import choice
import copy
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets, Dataset

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
    
    def __init__(self, data_dir, split, dense_features):

        self.unseen = True
        if split == 'seen':
            self.unseen = False
            with open(data_dir+'data_info.json') as f:
                feature_values = json.loads(f.read()).keys()
        else:
            with open(data_dir+'data_info.json') as f:
                feature_values = ['unseen']
            
        self.dataset = data_dir.split('/')[-2]
        self.datasets = []
        self.lengths = {'start':[],'current':[]}
            
        def _load_dataset(data_file, split, dense_features):
            dataset = load_dataset('json', data_files=data_file, streaming=True)['train']
            dataset_length = sum(1 for e in open(data_file,'r'))
            if self.dataset == 'movie_dialogue' or dense_features:
                train_samples = 10000
            else:
                train_samples = 100000
            length = train_samples/0.9
            train_length = floor(length*0.9)
            val_length = floor(length*0.05)
            test_length = floor(length*0.05)
            if split == 'seen':
                assert (train_length + val_length + test_length) < dataset_length
                return dataset.skip(train_length+val_length).take(test_length), test_length
            else:
                assert (test_length) < dataset_length
                return dataset.take(test_length), test_length

        for feature_value in list(feature_values):
            if dense_features:
                dataset, length = _load_dataset(data_dir+'files/'+feature_value+'_data_filtered.json', split, dense_features)
            else:
                dataset, length = _load_dataset(data_dir+'files/'+feature_value+'_data.json', split, dense_features)
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
                    if self.unseen:
                        raw_item[key] = [0]
                    else:
                        raw_item[key] = [i for i in raw_item[key] if i in [1, 3, 6, 7, 9, 14, 17, 18, 19, 22]]
                    item[key] = random.choice(raw_item[key])
                else:
                    item[key] = raw_item[key]
            elif key == 'sentence':
                item['input_ids'] = raw_item[key][:MAX_LENGTH]
                item['attention_mask'] = [1]*len(item['input_ids'])
        return item
        
    def __len__(self):
        return sum(self.lengths['start'])
    


def test(config: DictConfig) -> Optional[float]:

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning model
    print(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.ckpt)
    model.to('cuda')
    model.eval()
    torch.set_grad_enabled(False)
    
    for split in ['unseen','seen']:
        if not (config.datamodule.dense_features and split == 'unseen'):
            # Init lightning datamodule
            print(f"Instantiating datamodule <{config.datamodule._target_}>")
            data_test = LanguageDataset(
                config.datamodule.data_dir+'/'+config.datamodule.dataset+'/',
                split, config.datamodule.dense_features)
            test_dataloader = DataLoader(dataset=data_test,
                                         collate_fn=collate_fn,
                                         batch_size=1,
                                         num_workers=1,
                                         pin_memory=config.datamodule.pin_memory,
                                         shuffle=False)

            print('Testing!')
            output = {'loss':[]}
            for i,batch in tqdm(enumerate(test_dataloader)):

                if i == 0:
                    for key in batch.keys():
                        if key != 'attention_mask':
                            output[key] = []

                # train step
                losses = model.test_step(batch,i)

                for key in batch.keys():
                    if key != 'attention_mask':
                        output[key].append(batch[key][0])
                output['loss'].append(losses['loss'].item())

            pd.DataFrame(output).to_json('/'.join(config.ckpt.split('/')[:-2])+'/'+split+'.json')
            print('Saving results to '+'/'.join(config.ckpt.split('/')[:-2])+'/'+split+'.json')
            
    return
