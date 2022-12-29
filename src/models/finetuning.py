from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

import copy
import random
from tqdm import tqdm
from transformers import AdamW, GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

import json
with open('../../../../data/features.txt') as f:
    FEATURES = json.loads(f.read())
    
class FineTuningModel(LightningModule):

    def __init__(
        self,
        dataset: str = 'movie_dialogue',
        feature: str = 'genre',
        model: str = 'standard',
        max_feature_values: int = 10000,
        original_tokens: bool = True,
        lr: float = 0.00001,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # initialize model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if model == 'conditional':
            if not original_tokens:
                with open('../../../../data/dense_features.txt') as f:
                    self.tokens = json.loads(f.read())
                self.features = list(self.tokens[dataset].keys())
            else:
                self.features = list(FEATURES[dataset].keys())
            if self.hparams.feature != 'all':
                self.features = [self.hparams.feature]
            special_tokens_dict = {'additional_special_tokens': ['<G>']}
            print('adding feature tokens...')
            for feature in self.features:
                special_tokens_dict['additional_special_tokens'].append('<'+feature+'>')
            print('adding feature value tokens...')
            for feature_value in tqdm(range(max_feature_values)):
                special_tokens_dict['additional_special_tokens'].append('<'+str(feature_value)+'>')
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.sep_tok = self.tokenizer.encode('<G>')[0]
            self.model.resize_token_embeddings(len(self.tokenizer))

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, train: bool):
        if self.hparams.model == 'standard':
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
            labels = input_ids
        else:
            batch_size = input_ids.shape[0]
            prompt = []
            for batch_idx in range(batch_size):
                prompt_ids = []
                base_id = 1
                for feature in self.features:
                    substitute_unknown = random.choice(range(10)) == 0 if train else False
                    if self.hparams.original_tokens:
                        base_id = (self.features.index(feature)+1)*10000000
                        if not substitute_unknown and (self.hparams.feature in ['all',feature]):
                            feature_value = str(self.meta[feature][batch_idx]+base_id)
                        else:
                            feature_value = str(base_id)
                        while len(feature_value)!=8:
                            feature_value = '0'+feature_value
                        feature_value = [int(feature_value[:4]), int(feature_value[4:])]
                    else:
                        if not substitute_unknown and (self.hparams.feature in ['all',feature]):
                            if self.meta[feature][batch_idx] == 0:
                                feature_value = [base_id]
                            else:
                                feature_value = [self.tokens[self.hparams.dataset][feature].index(self.meta[feature][batch_idx]) + 1 + base_id]
                        else:
                            feature_value = [base_id]
                        base_id += len(self.tokens[self.hparams.dataset][feature]) + 1                        
                    prompt_ids += [feature_value]
                if self.hparams.original_tokens:
                    prompt.append(''.join(['<'+f+'> <'+str(fv[0])+'> <'+str(fv[1])+'> ' for f, fv in zip(self.features,prompt_ids)])+'<G>')
                else:
                    prompt.append(''.join(['<'+f+'> <'+str(fv[0])+'> ' for f, fv in zip(self.features,prompt_ids)])+'<G>')
                    
            prompt = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
            input_ids = torch.cat([prompt['input_ids'], input_ids[:,1:]],dim=1).to(self.model.device)
            attention_mask = torch.cat([prompt['attention_mask'], attention_mask[:,1:]],dim=1).to(self.model.device)
            labels = copy.deepcopy(input_ids)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            idx = len(prompt['input_ids'][0])-1
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx+1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, logits, labels

    def step(self, batch: Any, train: bool):
        self.meta = {key:batch[key+'_id'] for key in FEATURES[self.hparams.dataset].keys()}
        input_ids = torch.tensor(batch['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(self.model.device)
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=input_ids, train=train)
        loss = outputs[0]
        preds = torch.argmax(outputs[1], dim=-1)
        labels = outputs[2]
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
        params = self.model.parameters()
        return AdamW(params, lr=self.hparams.lr)