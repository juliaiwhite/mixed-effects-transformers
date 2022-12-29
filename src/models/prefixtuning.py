from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

import copy
import random
from transformers import AdamW, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

import json
with open('../../../../data/features.txt') as f:
    FEATURES = json.loads(f.read())

class PrefixTuningModel(LightningModule):

    def __init__(
        self,
        dataset: str = 'movie_dialogue',
        feature: str = 'genre',
        model: str = 'standard',
        original_tokens: bool = True,
        dropout: bool = True,
        prefix_size: int = 10000,
        num_prefixes: int = 2,
        lr: float = 0.00001,
        beta: float = 0.0001,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # initialize model
        self.config = GPT2Config.from_pretrained('gpt2')
        self.n_embd_per_head = self.config.n_embd // self.config.n_head
        
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bos_id = torch.tensor(self.tokenizer(self.tokenizer.bos_token)['input_ids']).unsqueeze(0)
        
        if not original_tokens:
            with open('../../../../data/dense_features.txt') as f:
                self.tokens = json.loads(f.read())
            self.features = list(self.tokens[dataset].keys())
        else:
            self.features = list(FEATURES[dataset].keys())
        
        self.prefix_len = num_prefixes*(len(self.features)+1)
        self.prefix_size = prefix_size

        mid_dim = 512
        if model == 'seperate':
            self.prefix_embed_0 = nn.Embedding(self.prefix_size, self.config.n_embd)
            self.prefix_mlp_0 = nn.Sequential(nn.Linear(self.config.n_embd, mid_dim),
                                            nn.Tanh(),
                                            nn.Linear(mid_dim, self.config.n_embd*2*self.config.n_layer))
            self.prefix_embed_1 = nn.Embedding(self.prefix_size, self.config.n_embd)
            self.prefix_mlp_1 = nn.Sequential(nn.Linear(self.config.n_embd, mid_dim),
                                            nn.Tanh(),
                                            nn.Linear(mid_dim, self.config.n_embd*2*self.config.n_layer))
        else:
            self.prefix_embed = nn.Embedding(self.prefix_size, self.config.n_embd)
            self.prefix_mlp = nn.Sequential(nn.Linear(self.config.n_embd, mid_dim),
                                            nn.Tanh(),
                                            nn.Linear(mid_dim, self.config.n_embd*2*self.config.n_layer))
       
        if dropout:
            self.dropout = nn.Dropout(0.1)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        
    def train(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        if self.hparams.model == 'seperate':
            self.prefix_embed_0.train()
            self.prefix_mlp_0.train()
            self.prefix_embed_1.train()
            self.prefix_mlp_1.train()
        else:
            self.prefix_embed.train()
            self.prefix_mlp.train()
        if self.hparams.dropout:
            self.dropout.train()
        
    def eval(self):
        self.model.eval()
        if self.hparams.model == 'seperate':
            self.prefix_embed_0.eval()
            self.prefix_mlp_0.eval()
            self.prefix_embed_1.eval()
            self.prefix_mlp_1.eval()
        else:
            self.prefix_embed.eval()
            self.prefix_mlp.eval()
        if self.hparams.dropout:
            self.dropout.eval()

    def get_prompt(self, batch_size: int, get_unknown: bool = False):
        if self.hparams.model == 'standard':
            prefix_raw = torch.zeros([batch_size,self.prefix_len]).long().to(self.model.device)#torch.randint(self.prefix_size,[batch_size,self.prefix_len]).to(self.model.device)
        else:
            prefix_raw = []
            for batch_idx in range(batch_size):
                prefix_tokens = [0]*self.hparams.num_prefixes
                base_id = 1
                for feature in self.features:
                    if self.hparams.original_tokens:
                        base_id = (self.features.index(feature)+1)*10000000
                        if not get_unknown and (self.hparams.feature in ['all',feature]):
                            feature_value = str(self.meta[feature][batch_idx]+base_id)
                        else:
                            feature_value = str(base_id)
                        while len(feature_value)!=8:
                            feature_value = '0'+feature_value
                        prefix_tokens += [int(feature_value[:4]), int(feature_value[4:])]
                    else:
                        if not get_unknown and (self.hparams.feature in ['all',feature]):
                            if self.meta[feature][batch_idx] == 0:
                                prefix_tokens += [base_id]*self.hparams.num_prefixes
                            else:
                                prefix_tokens += [self.tokens[self.hparams.dataset][feature].index(self.meta[feature][batch_idx]) + 1 + base_id]*self.hparams.num_prefixes
                        else:
                            prefix_tokens += [base_id]*self.hparams.num_prefixes
                        base_id += len(self.tokens[self.hparams.dataset][feature]) + 1
                prefix_raw.append(prefix_tokens)
            prefix_raw = torch.tensor(prefix_raw).to(self.model.device)
            
        if self.hparams.model == 'seperate':
            prefixes = []
            for idx in range(self.hparams.num_prefixes):
                prefix_raw = prefix_raw[:,idx*(len(self.features)+1):(idx+1)*(len(self.features)+1)]
                prefix_hidden = self.prefix_embed_0(prefix_raw)
                prefixes.append(self.prefix_mlp_0(prefix_hidden))
                prefixes[-1] = prefixes[-1].view(batch_size, 
                                                 (len(self.features)+1), 
                                                 self.config.n_layer*2, 
                                                 self.config.n_head,
                                                 self.n_embd_per_head)
            prefix = torch.cat(prefixes,dim=1)
        else:
            prefix_hidden = self.prefix_embed(prefix_raw)
            prefix = self.prefix_mlp(prefix_hidden)
            prefix = prefix.view(batch_size, 
                                 self.prefix_len, 
                                 self.config.n_layer*2, 
                                 self.config.n_head,
                                 self.n_embd_per_head)
            
        if self.hparams.dropout:
            prefix = self.dropout(prefix)
            
        return prefix.permute([2,0,3,1,4]).to(self.model.device).split(2)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, train: bool):
        get_unknown = random.choice(range(10)) == 0 if train else False
        past_key_values = self.get_prompt(batch_size=input_ids.shape[0], get_unknown=get_unknown)
        input_ids = torch.stack(list(input_ids), dim=0)
        attention_mask = torch.stack(list(attention_mask), dim=0)
        padded_mask = torch.ones(attention_mask.shape[0],
                                 attention_mask.shape[1]+self.prefix_len).to(self.model.device)
        padded_mask[:,self.prefix_len:] = attention_mask
        
        if get_unknown or not train:
            regularization_term = 0
        else:
            unknown_past_key_values = self.get_prompt(batch_size=input_ids.shape[0], get_unknown=True)
            regularization_term = ((torch.stack(unknown_past_key_values,dim=0)-torch.stack(past_key_values,dim=0))**2).sum()
        
        outputs = self.model(input_ids=input_ids, attention_mask=padded_mask, past_key_values=past_key_values, labels=labels)
        loss = outputs[0] + self.hparams.beta*regularization_term
        preds = torch.argmax(outputs[1], dim=-1)
        return loss, preds
    
    def step(self, batch: Any, train: bool):
        self.meta = {key:batch[key+'_id'] for key in FEATURES[self.hparams.dataset].keys()}
        input_ids = torch.tensor(batch['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(self.model.device)
        loss, preds = self.forward(input_ids, attention_mask=attention_mask, labels=input_ids, train=train)
        labels = input_ids
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, train=True)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(batch['input_ids']))
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch['input_ids']))

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, train=True)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(batch['input_ids']))
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch['input_ids']))

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, train=False)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=len(batch['input_ids']))
        self.log("test/acc", acc, on_step=False, on_epoch=True, batch_size=len(batch['input_ids']))

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        if self.hparams.model == 'seperate':
            params = list(self.prefix_embed_0.parameters()) + list(self.prefix_mlp_0.parameters()) + list(self.prefix_embed_1.parameters()) + list(self.prefix_mlp_1.parameters())
        else:
            params = list(self.prefix_embed.parameters()) + list(self.prefix_mlp.parameters())
        return AdamW(params, lr=self.hparams.lr)