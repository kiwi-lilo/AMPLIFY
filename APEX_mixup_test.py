
import sys
import argparse
import gc
import copy
import os
import os.path as osp
import traceback
import inspect
import socket
import types
import yaml
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = 'ca50793e93406ee2a660b1dc15665294cd1b9e70'
os.environ["TRANSFORMERS_OFFLINE"] = '1'
os.environ["HF_DATASETS_OFFLINE"] = '1'
import GPUtil
if GPUtil.getGPUs()[0].name == 'NVIDIA GeForce RTX 3080 Ti':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

sys.path.append(os.getcwd())

import yagmail
import datasets
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from datetime import datetime
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from collections import OrderedDict
import wandb
from prefetch_generator import BackgroundGenerator
from torchnlp.encoders import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score,log_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy ,F1Score

import torch

import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer.supporters import CombinedLoader

from pytorch_lightning.loggers import WandbLogger,MLFlowLogger,CSVLogger #CometLogger
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping ,Timer,DeviceStatsMonitor ,StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loops.base import Loop
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything ,Callback
# from pytorch_lightning.profiler import AdvancedProfiler ,PyTorchProfiler

from datasets import load_dataset
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable

import warnings
from loguru import logger
from tqdm import tqdm, trange
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DebertaV2Config,
    DebertaV2Tokenizer,
    BertTokenizer,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    RoFormerTokenizer,
    RoFormerModel, 
    RoFormerConfig,
    RoFormerForSequenceClassification,

    # NezhaConfig,
    # NezhaModel,
    # NezhaForSequenceClassification
)

from mixup_model.BertModelm import BertModel

import transformers
transformers.logging.set_verbosity_error()


timer_callback = Timer()
from sklearn.linear_model import LogisticRegression

from scipy.special import softmax
from scipy.optimize import nnls
import scipy.stats
import spacy
# from pytorch_lightning.loops.fit_loop import FitLoop
# from pytorch_lightning.trainer.states import TrainerFn

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    logger.info(tb)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class DataModule(LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]
    task_test_text_field_map = {
        "cola": ["text"],
        "imdb": ["sentence"],
        "ChnSentiCorp": ["sentence"],
        "TNEWS": ["content"],
        "iflytek": ['sentence'],
        "DBPedia": ["sentence1", "sentence2"],
        "AG_news": ["sentence"],
        "yelp_2": ["sentence"],
        "yelp_5": ["sentence"],
        "SST-2": ["sentence"],
        "SST-5": ["sentence"],
        "mrpc": ["text1", "text2"],
        "trec": ["sentence"],
        "trec_coarse": ["sentence"],
        "trec_fine": ["sentence"],
        "rte": ["text1", "text2"],
        "qnli": ["text1", "text2"],
        "mnli": ["text1", "text2"],
        "qqp": ["text1", "text2"],
        "emotion": ["text"],
        "civilcomments":["text"],
        "tweet_eval_emoji":["text"],
        "CR": ["text"],
        "subj":["text"],
    }
    task_train_text_field_map = {
        "cola": ["text"],
        "imdb": ["sentence"],
        "ChnSentiCorp": ["content"],
        "TNEWS": ["content"],
        "iflytek": ['sentence'],
        "DBPedia": ["sentence1", "sentence2"],
        "AG_news": ["sentence"],
        "yelp_2": ["sentence"],
        "yelp_5": ["sentence"],
        "SST-2": ["sentence"],
        "SST-5": ["sentence"],
        "mrpc": ["text1", "text2"],
        "trec": ["sentence"],
        "trec_coarse": ["sentence"],
        "trec_fine": ["sentence"],
        "rte": ["text1", "text2"],
        "qnli": ["text1", "text2"],
        "mnli": ["text1", "text2"],
        "qqp": ["text1", "text2"],
        "emotion": ["text"],
        "civilcomments":["text"],
        "tweet_eval_emoji":["text"],
        "CR": ["text"],
        "subj":["text"],

    }
    task_split_field_map = {
        "ChnSentiCorp": ["train","test"],
        "TNEWS": ["train","test"],
        "iflytek": ["train","test"],

        "yelp_2": ["train","test"],
        "yelp_5": ["train","test"],
        "DBPedia": ["train","test"],
        "cola": ["train","test"],
        
        "trec_fine": ["train","test"],
        "trec_coarse": ["train","test"],
        "AG_news": ["train","test"],
        "imdb": ["train","test"],

        "SST-2": ["train","validation","test"],
        "SST-5": ["train","validation","test"],
        "mrpc": ["train","validation","test"],
        "emotion": ["train","validation","test"],
        "civilcomments":["train","validation","test"],
        "tweet_eval_emoji":["train","validation","test"],

        "CR":["train","test"],
        "subj":["train","test"],

        "rte": ["train","validation"], #796k test unlabel
        "qnli": ["train","validation"], #23m test unlabel
        "mnli": ["train","validation"], #66m test unlabel
        "qqp": ["train","validation"], #43m test unlabel
        
    }


    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        max_seq_length: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        tta_type: list,
        seed: int,
        train_augnum: int,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset = {}
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_name_or_path}", use_fast=True)
        if self.model_name_or_path == "roberta_base":
            self.tokenizer = AutoTokenizer.from_pretrained('roberta_base', add_prefix_space=True, use_fast=True)
        
        self.all_csv = {}
        self.all_split = self.task_split_field_map[self.task_name]
        for split in self.all_split:
            self.all_csv[split] = pd.read_csv(f"datasets/{self.task_name}/{split}.csv")
        if train_augnum > 0:
            self.all_csv["train"] = pd.read_csv(f"datasets/{self.task_name}/tta_eda_0.1_1+{train_augnum}_train.csv")
        
        self.label_encoder = LabelEncoder(
            self.all_csv['train'].label.astype(str).unique().tolist(),
            reserved_labels=[],
        )
        self.seed= seed
        self.tta_type = tta_type
        self.num_labels = len(self.label_encoder.index_to_token)
        self.text_fields = self.task_train_text_field_map[self.task_name]

    def setup(self, stage: str):

        if 'validation' not in self.all_split:
            train_df, val_df = train_test_split(self.all_csv['train'], test_size=0.25, random_state=self.seed)
            self.all_csv['train'] = train_df
            self.all_csv['validation'] = val_df
            self.all_csv['test'] = pd.read_csv(f"datasets/{self.task_name}/test.csv")

        elif 'test' not in self.all_split:
            train_df, val_df = train_test_split(self.all_csv['train'], test_size=0.25, random_state=self.seed)
            self.all_csv['train'] = train_df
            self.all_csv['validation'] = val_df
            self.all_csv['test'] = pd.read_csv(f"datasets/{self.task_name}/validation.csv")
    
        if stage == "fit":
            for split in ['train', 'validation', 'test']:
                self.dataset[split] = datasets.Dataset.from_pandas(self.all_csv[split])
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=['label']
                )
                self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
                self.dataset[split].set_format(
                    type="torch",
                    columns=self.columns
                )

    def train_dataloader(self):
        # drop_last = True if len(self.train_fold) > self.train_batch_size else False
        return DataLoaderX(
            self.dataset["train"],
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            pin_memory=False,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoaderX(
            self.dataset["validation"],
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            pin_memory=False,
            drop_last=False
        )

    def predict_dataloader(self):
        return DataLoaderX(
            self.dataset["test"],
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            pin_memory=False,
            drop_last=False
        )

    def convert_to_features(self, example_batch, indices=None):
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=False
        )
        if 'label' in example_batch.keys():
            features["labels"] = [int(self.label_encoder.index_to_token.index(str(i))) for i in example_batch['label']]

        return features

    def __post_init__(cls):
        super().__init__()


class pretraining_model(nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 forward_type: str,
                 num_labels : int,
        ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.config = AutoConfig.from_pretrained(f"{self.model_name_or_path}", output_hidden_states=True)

        if 'roberta_base' in self.model_name_or_path:
            self.model = RobertaForSequenceClassification.from_pretrained(f"{self.model_name_or_path}", num_labels=num_labels)

        elif 'bert_base_uncased' in self.model_name_or_path:
            self.model = AutoModel.from_pretrained(f"{self.model_name_or_path}", num_labels=num_labels)
           
        elif 'distilbert-base-uncased' in self.model_name_or_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(f"{self.model_name_or_path}", num_labels=num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.forward_type = forward_type
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self._init_weights(self.classifier)
        # for param in self.model.parameters():
        #     param.requires_grad_(True) #是否进行梯度更新

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def forward(self, inputs,labels=None):
        
    #     input_ids = inputs['input_ids']
    #     attention_mask = inputs['attention_mask']
    #     # token_type_ids = inputs['token_type_ids']
    #     outputs = self.model(
    #         input_ids, 
    #         attention_mask=attention_mask, 
    #         # token_type_ids=token_type_ids,
    #         labels=labels
    #     )
    #     return outputs
    def loss(self, logits, labels):
        loss_fnc = nn.CrossEntropyLoss(reduction='mean')
        # loss_fnc = DiceLoss(smooth = 1, square_denominator = True, with_logits = True, alpha = 0.00011 )
        loss = loss_fnc(logits, labels)
        return loss

    def forward(self, inputs, labels=None):
        outputs = self.model(**inputs)
        feature = torch.mean(outputs[0], axis=1)
        # feature = outputs[1]
        feature = self.dropout(feature)
        features = self.classifier(feature)
        features = F.log_softmax(features, dim=1)

        _loss = 0
        if labels is not None:
            _loss = self.loss(features,labels)
        return features, _loss


class TrainingModule(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        encoder_lr: float = 2e-5,
        decoder_lr: float = 2e-5,
        # learning_rate: float = 2e-5,
        correct_biased: bool = False,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        forward_type : str = "normal",
        freeze_encoder: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()


        self.model_name_or_path = model_name_or_path
        self.model = pretraining_model(
            model_name_or_path = self.model_name_or_path,
            forward_type = forward_type,
            num_labels = num_labels
        )
        self.forward_type = forward_type
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder == True:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:2])

            freezed_parameters = get_freezed_parameters(self.model)
            print(f"Freezed parameters: {freezed_parameters}")

        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_labels = num_labels
        self.adam_epsilon = adam_epsilon
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        # self.learning_rate = learning_rate
        self.correct_biased = correct_biased

    def training_step(self, batch, batch_idx):
        # labels = batch['labels']
        # inputs = {k: v for k, v in batch.items() if k != 'labels'}
        # output = self.model(inputs,labels)
        # loss = output.loss
        # logits = output.logits
        # return loss
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        logits, loss = self.model(inputs, labels)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # labels = batch['labels']
        # inputs = {k: v for k, v in batch.items() if k != 'labels'}
        # output = self.model(inputs,labels)
        # loss = output.loss
        # logits = output.logits
        # return {"loss": loss, "preds": logits, "labels": labels}
        labels = batch['labels']
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        logits, val_loss = self.model(inputs, labels)
        return {"loss": val_loss, "preds": logits, "labels": labels}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # labels = batch['labels']
        # inputs = {k: v for k, v in batch.items() if k != 'labels'}
        # output = self.model(inputs)
        # logits = output.logits
        # return {"logits": logits}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        logits, _ = self.model(inputs)
        return {"logits": logits}
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # inputs = {k: v for k, v in batch.items() if k != 'labels'}
        # output = self.model(inputs)
        # logits = output.logits
        # return {"logits": logits}
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        logits, _ = self.model(inputs)
        return {"logits": logits}

    def validation_epoch_end(self, outputs):
        # print('validation_epoch_end')
        # batch_pred = torch.cat([x["preds"] for x in outputs])
        # batch_true = torch.cat([x["labels"] for x in outputs])
        # loss = torch.stack([x["loss"] for x in outputs]).mean()

        # valid_pred = [item.argmax(axis=-1) for item in batch_pred.detach().cpu().numpy()]
        # valid_true = [item for item in batch_true.detach().cpu().numpy()]

        # val_acc = accuracy_score(valid_true, valid_pred)
        # val_f1 = f1_score(valid_true, valid_pred, average='macro')
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", val_acc, prog_bar=True)
        # self.log("val_f1", val_f1, prog_bar=True)
        # return loss
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        pred = [item.argmax(axis=-1) for item in preds.detach().cpu().numpy()]
        label = [item for item in labels.detach().cpu().numpy()]
        val_acc = accuracy_score(label, pred)
        val_f1 = f1_score(label, pred, average='macro')
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.hparams.decoder_lr,
                "weight_decay": self.weight_decay,
                'initial_lr':self.hparams.encoder_lr,
                'correct_biased_decay': self.hparams.correct_biased
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.hparams.decoder_lr,
                "weight_decay": 0.0,
                'initial_lr':self.hparams.encoder_lr,
                'correct_biased_decay': self.hparams.correct_biased
            }
        ]
        if self.freeze_encoder == True:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p.requires_grad in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'lr': self.hparams.decoder_lr,
                    "weight_decay": self.weight_decay,
                    'initial_lr':self.hparams.encoder_lr,
                    'correct_biased_decay': self.hparams.correct_biased
                },
                {
                    "params": [p for n, p.requires_grad in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    'lr': self.hparams.decoder_lr,
                    "weight_decay": 0.0,
                    'initial_lr':self.hparams.encoder_lr,
                    'correct_biased_decay': self.hparams.correct_biased
                }
            ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.hparams.adam_epsilon,
            # lr=self.hparams.learning_rate,
        )
        if self.hparams.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        if self.hparams.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.hparams.num_cycles
            )

        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]


def class2dict(obj):
    d = {}
    d['__class__'] = obj.__class__.__name__
    d['__module__'] = obj.__module__
    d.update(obj.__dict__)
    return d


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("[Run Validation]")
        return bar

def freeze(module):
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """

    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters

def get_f1_acc(true, pred):
    f1 = f1_score(
        true,
        pred,
        average='macro'
    )
    acc = accuracy_score(
        true,
        pred,
    )
    return acc, f1

def send_mail(title,content):
    try:
        yag = yagmail.SMTP(user="353939483@qq.com",password="vejanjaqzcfgbgfb", host='smtp.qq.com')
        yag.send(
            to=["782319269@qq.com"],
            subject=title,
            contents=content,
        )
    except Exception as e:
        pass

def clear_all_cache():
    try:
        # clear torch cache
        torch.cuda.empty_cache()
        # clear all global variables
        for key, value in globals().items():
            if callable(value) or value.__class__.__name__ == "module":
                continue
            del globals()[key] 
        # collect garbage
        gc.collect() 
    except Exception as e:
        print(e)
        pass

def find_best_f1_epoch(checkpoint_path):
    best_epoch = 0
    best_f1 = 0
    best_f1_epoch_file_name = ''
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            f1 = float(file.split('_')[3])
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = int(file.split('_')[1])
                best_f1_epoch_file_name = file
            
    return best_epoch, best_f1_epoch_file_name

def find_last_epoch(checkpoint_path):
    last_epoch = 0
    last_epoch_file_name = ''
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            epoch = int(file.split('_')[1])
            if epoch > last_epoch:
                last_epoch = epoch
                last_epoch_file_name = file
    return last_epoch, last_epoch_file_name

def write_basic_log(config, checkpoint_path):
    basic_log=dict() 
    basic_log['code']=inspect.getsource(sys.modules[__name__])
    basic_log['pid']=os.getpid()
    basic_log['datetime']=str(datetime.now())
    basic_log['hostname']=socket.gethostname()
    basic_log['cwd']=os.getcwd()
    basic_log['python_version']=sys.version
    basic_log['python_executable']=sys.executable
    basic_log['python_prefix']=sys.prefix
    basic_log['python_platform']=sys.platform
    basic_log['GPU']=torch.cuda.get_device_name(0)
    with open(f'{checkpoint_path}/basic.log', 'w') as f:
        f.write(yaml.dump(basic_log, Dumper=yaml.CDumper))

@hydra.main(version_base=None, config_path="configs",config_name="base_config")
def main(cfg: DictConfig) -> None:
    log_time=time.strftime("%Y%m%d%H-%M", time.localtime())
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    Config = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    root_dir = f"{Config.basic.seed}_{Config.basic.root_dbl}_mixup"
    Config.basic.logging_dir = f'{root_dir}/{Config.basic.task_name}/log/{Config.model.model_name}/wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'              
    Config.basic.checkpoint_path = f'{root_dir}/{Config.basic.task_name}/models/{Config.model.model_name}/wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'       
    Config.basic.profiler_path = f'{root_dir}/{Config.basic.task_name}/profiler/{Config.model.model_name}/wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'
    if Config.trainer.auto_lr_find == True and Config.trainer.use_swa == False:
        Config.basic.logging_dir = f'{root_dir}/{Config.basic.task_name}/log/{Config.model.model_name}/lr_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'              
        Config.basic.checkpoint_path = f'{root_dir}/{Config.basic.task_name}/models/{Config.model.model_name}/lr_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'       
        Config.basic.profiler_path = f'{root_dir}/{Config.basic.task_name}/profiler/{Config.model.model_name}/lr_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'
    elif Config.trainer.use_swa == True and Config.trainer.auto_lr_find == False:
        Config.basic.logging_dir = f'{root_dir}/{Config.basic.task_name}/log/{Config.model.model_name}/swa_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'              
        Config.basic.checkpoint_path = f'{root_dir}/{Config.basic.task_name}/models/{Config.model.model_name}/swa_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'       
        Config.basic.profiler_path = f'{root_dir}/{Config.basic.task_name}/profiler/{Config.model.model_name}/swa_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'
    elif Config.trainer.use_swa == True and Config.trainer.auto_lr_find == True:
        Config.basic.logging_dir = f'{root_dir}/{Config.basic.task_name}/log/{Config.model.model_name}/lr_swa_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'              
        Config.basic.checkpoint_path = f'{root_dir}/{Config.basic.task_name}/models/{Config.model.model_name}/lr_swa_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'       
        Config.basic.profiler_path = f'{root_dir}/{Config.basic.task_name}/profiler/{Config.model.model_name}/lr_swa_wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{Config.trainer.encoder_lr}'

    # ------------------------
    # SEED
    # ------------------------
    seed_everything(Config.basic.seed)
    try:
        if not os.path.exists(f"{root_dir}/TTA"):
            os.makedirs(f"{root_dir}/TTA")
        if not os.path.exists(f"{root_dir}/label_change"):
            os.makedirs(f"{root_dir}/label_change")
        if not os.path.exists(Config.basic.checkpoint_path):
            os.makedirs(Config.basic.checkpoint_path)
        if not os.path.exists(f"{Config.basic.logging_dir}"):
            os.makedirs(Config.basic.logging_dir)
        if not os.path.exists(f"{Config.basic.profiler_path}"):
            os.makedirs(Config.basic.profiler_path)
        write_basic_log(Config, Config.basic.checkpoint_path)


        # ------------------------
        # SET LOGGER
        # ------------------------
        logger.info(Config)
        logger.info('===' * 20)
        logger.info(f"{Config.basic.checkpoint_path}")
        logger.info('===' * 20)
        # ---------------------
        # RUN TRAINING
        # ---------------------

        # ========================
        # DataModule
        # ========================
        dm = DataModule(
            model_name_or_path=f"models/{Config.model.model_name}",
            task_name=Config.basic.task_name,
            max_seq_length=Config.model.max_seq_length,
            train_batch_size=Config.trainer.train_batch_size,
            eval_batch_size=Config.trainer.eval_batch_size,
            num_workers=Config.data.num_workers,
            tta_type=Config.basic.tta_type,
            seed=Config.basic.seed,
            train_augnum=Config.data.train_augnum,
        )
        # ---------------------
        # MODEL
        # ---------------------
        model = TrainingModule(
            model_name_or_path=f"models/{Config.model.model_name}",
            num_labels=dm.num_labels,
            task_name=Config.basic.task_name,
            batch_size=Config.trainer.train_batch_size,
            encoder_lr=Config.trainer.encoder_lr,
            decoder_lr=Config.trainer.decoder_lr,
            # learning_rate=Config.trainer.encoder_lr,
            correct_biased=Config.model.correct_biased,
            num_cycles=Config.model.num_cycles,
            scheduler=Config.model.scheduler,
            forward_type=Config.model.forward_type,
            freeze_encoder=Config.model.freeze_encoder,
            weight_decay = Config.model.weight_decay,
            warmup_steps = Config.model.num_warmup_steps,
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor="val_f1",
            dirpath=f'{Config.basic.checkpoint_path}',
            filename= "{Config.basic.task_name}_{epoch}_{val_acc:.6f}_{val_f1:.6f}_{Config.trainer.limit_train_batches}_",
            auto_insert_metric_name=False,
            mode='max',
            save_last=False,
            verbose=True
        )
        earlyStopping_callback = EarlyStopping(
            monitor='val_f1',
            patience=5,
            mode='max'
        )
        csvlogger_logger = CSVLogger(
            save_dir=f'{Config.basic.logging_dir}',
            name=f'{Config.model.model_name}/wm_{Config.model.num_warmup_steps}_wd_{Config.model.weight_decay}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.trainer.limit_train_batches}_{Config.model.forward_type}_{log_time}',
        )
        # comet_logger = CometLogger(
        #     api_key='akEFPqLQrYJi8HTuHkNZ6owVG',
        #     workspace="my20889938", 
        #     save_dir=f'{Config.basic.logging_dir}', 
        #     project_name="nlp-tta", 
        #     experiment_name=f'{Config.basic.checkpoint_path}',
        # )
        # comet_logger.log_graph(model.model)
        # comet_logger.log_hyperparams(class2dict(Config))

        #wandb.init(
        #   name=f'{Config.basic.task_name}_{Config.trainer.limit_train_batches}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.model.model_name}',
        #)
        #wandb_logger = WandbLogger(
        #     project="TTTA"
        #)
        # wandb_logger.experiment.config.update(class2dict(Config))

        #wandb_logger.watch(model, log="all")
        # profiler = PyTorchProfiler(
            # dirpath=f"{Config.basic.profiler_path}", 
            # filename="perf_logs"
        # )
        trainer_callbacks = [
            checkpoint_callback,
            earlyStopping_callback,
            # TQDMProgressBar(refresh_rate = 10)
            LitProgressBar(),
            timer_callback,
            DeviceStatsMonitor()
        ]

        if Config.trainer.use_swa == True:
            trainer_callbacks.append(
                StochasticWeightAveraging()
            )
        # ---------------------
        # TRAINER
        # ---------------------
        trainer = Trainer(
            max_epochs = Config.trainer.max_epochs,
            devices = 1,
            # devices = [0],
            auto_select_gpus = False,
            accelerator = "gpu",
            auto_scale_batch_size = None,
            accumulate_grad_batches = Config.trainer.accumulate_grad_batches,
            check_val_every_n_epoch = Config.trainer.check_val_every_n_epoch,
            amp_backend = Config.trainer.amp_backend,
            precision = Config.trainer.mixed_precision,
            deterministic = Config.trainer.deterministic,
            benchmark = Config.trainer.benchmark,
            default_root_dir = os.getcwd(),
            enable_checkpointing = Config.trainer.enable_checkpointing,
            enable_progress_bar = True,
            logger = [
                # comet_logger,
                # wandb_logger,
                csvlogger_logger
            ],
            callbacks = trainer_callbacks,
            val_check_interval = Config.trainer.val_check_interval,
            num_sanity_val_steps = Config.trainer.num_sanity_val_steps, 
            limit_train_batches = Config.trainer.limit_train_batches, 
            limit_val_batches = Config.trainer.limit_val_batches,
            limit_test_batches = Config.trainer.limit_test_batches,
            limit_predict_batches = Config.trainer.limit_predict_batches,
            # profiler = profiler,
            auto_lr_find = Config.trainer.auto_lr_find,
        )
        logger.info(Config.basic.train_type)

        if Config.trainer.auto_lr_find == True:
            lr_finder = trainer.tuner.lr_find(
                model, 
                datamodule=dm,
                min_lr=1e-08,
                max_lr=1,
                num_training=100,
                mode='exponential',
                early_stop_threshold=4.0
            )
            fig = lr_finder.plot(suggest=True)
            fig.savefig(f"{Config.basic.logging_dir}/lr_finder.png")
            new_lr = lr_finder.suggestion()
            logger.info(f"New LR: {new_lr}")
            model.hparams.encoder_lr = new_lr
            model.hparams.decoder_lr = new_lr

            Config.trainer.encoder_lr = new_lr
            Config.trainer.decoder_lr = new_lr

            model = TrainingModule(
                model_name_or_path=f"models/{Config.model.model_name}",
                num_labels=dm.num_labels,
                task_name=Config.basic.task_name,
                batch_size=Config.trainer.train_batch_size,
                encoder_lr=new_lr,
                decoder_lr=new_lr,
                # learning_rate=new_lr,
                correct_biased=Config.model.correct_biased,
                weight_decay = Config.model.weight_decay,
                warmup_steps = Config.model.num_warmup_steps,
                num_cycles=Config.model.num_cycles,
                scheduler=Config.model.scheduler,
                forward_type=Config.model.forward_type,
            )

            # trainer.tune(
            #     model, 
            #     datamodule=dm,
                # scale_batch_size_kwargs = None,
                # lr_find_kwargs = {
                #     'mode': 'exponential',
                    # 'early_stop_threshold': 4.0,
                    # 'num_training': 100,
                    # 'min_lr': 1e-08,
            #         'max_lr': 1,
            #     }
            # )
        best_f1_epoch_file_name = ''
        if Config.trainer.resume_from_checkpoint == True:
            best_epoch,best_f1_epoch_file_name = find_best_f1_epoch(Config.basic.checkpoint_path)

        if 'train' in Config.basic.train_type:
            trainer.fit(
                model,
                ckpt_path = f"{Config.basic.checkpoint_path}/{best_f1_epoch_file_name}" if Config.trainer.resume_from_checkpoint == True else None,
                datamodule = dm,
            )

        if 'test' in Config.basic.train_type:
            best_epoch,best_f1_epoch_file_name = find_best_f1_epoch(Config.basic.checkpoint_path)
            last_eopch,last_epoch_file_name = find_last_epoch(Config.basic.checkpoint_path)

            if 'save_prediction' in Config.basic.train_type:
                # ---------------------
                #SAVE_PREDICTION
                # ---------------------
                dm.setup('fit')
                predict_checkpoint_path = ''
                if Config.basic.predict_from_checkpoint != "best":
                    predict_checkpoint_path=f"{Config.basic.checkpoint_path}/{Config.basic.predict_from_checkpoint}"
                    logger.info(f'load from specific checkpoint=> {predict_checkpoint_path}')
                elif Config.basic.predict_from_checkpoint == "best":
                    predict_checkpoint_path=f"{Config.basic.checkpoint_path}/{best_f1_epoch_file_name}"
                    logger.info(f'load from best checkpoint=> {predict_checkpoint_path}')
                elif Config.basic.predict_from_checkpoint == "last":
                    predict_checkpoint_path=f"{Config.basic.checkpoint_path}/{last_epoch_file_name}"
                    logger.info(f'load from last checkpoint=> {predict_checkpoint_path}')
                
                model = model.load_from_checkpoint(predict_checkpoint_path)
                model.eval()
                model.freeze()
                with torch.no_grad():
                    prediction_results = trainer.predict(
                        model,
                        dataloaders = dm.predict_dataloader(),
                        return_predictions=True
                    )
                    tta_y_pred = []
                    tta_y_pred_logits = []
                    for batch_preds in prediction_results:
                        tta_y_pred.extend(
                            batch_preds['logits'].argmax(-1).detach().cpu().numpy()
                        )
                        tta_y_pred_logits.extend(
                            batch_preds['logits'].detach().cpu().numpy()
                        )
                    np.save(f'{Config.basic.checkpoint_path}/{Config.basic.task_name}_{Config.trainer.limit_train_batches}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.model.model_name}_{Config.trainer.encoder_lr}_test_predictions.npy', tta_y_pred_logits)
                    #get final acc
                    tta_y_pred = np.array(tta_y_pred)
                    tta_y_true = np.array(dm.dataset["test"]['labels'])
                    acc = accuracy_score(tta_y_true, tta_y_pred)
                    f1 = f1_score(tta_y_true, tta_y_pred, average='macro')
                    logger.info(f"Final acc: {acc} f1: {f1}")
                    final_results = []
                    final_results.append({
                        'acc': acc,
                        'f1': f1,
                    })
                    final_results = pd.DataFrame(final_results)
                    final_results.to_csv(f'{root_dir}/TTA/{Config.basic.task_name}_{Config.trainer.limit_train_batches}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.model.model_name}_{Config.trainer.encoder_lr}_test_results.csv', index=False)

            # clear_all_cache() 

        send_mail(
            title= f'SUCCESS: {"_".join(Config.basic.train_type)} - {torch.cuda.get_device_name(0)}',
            content= f"{Config.basic.seed}_{Config.model.forward_type}_{Config.basic.task_name}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.model.model_name}_{Config.trainer.limit_train_batches}"
        )
    except Exception as e:
        print(e)
        send_mail(
            title= f'FAIL: {"_".join(Config.basic.train_type)} - {torch.cuda.get_device_name(0)}',
            content= f"{Config.basic.seed}_{Config.model.forward_type}_{Config.basic.task_name}_{Config.trainer.train_batch_size}_{Config.model.max_seq_length}_{Config.model.model_name}_{Config.trainer.limit_train_batches}_ERROR:{e}"
        )
        pass

if __name__ == "__main__":
    main()


