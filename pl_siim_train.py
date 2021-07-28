import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import pandas as pd
import numpy as np
import gc

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from tqdm.auto import tqdm
from functools import partial

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, ConcatDataset

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import torch.distributed as dist


# pytorch lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.config import *
from src.dataset import *
from src.models import *
from src.utils import *
from src.loss import *

import warnings 
warnings.filterwarnings('ignore')

from warmup_scheduler import GradualWarmupScheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]    

class SIIMPLModel(pl.LightningModule):
    def __init__(self,CFG, model, criterion):
        super(SIIMPLModel,self).__init__()
        self.CFG = CFG
        self.model = model
        self.train_criterion = criterion
        self.val_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    ### copy from https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    LOGGER.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                LOGGER.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)
    
    def get_scheduler(self,optimizer):
        if self.CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.CFG.factor, \
                    patience=self.CFG.patience, verbose=True, eps=self.CFG.eps)
        elif self.CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.CFG.T_max, eta_min=self.CFG.min_lr, last_epoch=-1)
        elif self.CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.CFG.T_0, T_mult=1, eta_min=self.CFG.min_lr, last_epoch=-1) 
        elif self.CFG.scheduler == 'GradualWarmupSchedulerV2':
            scheduler_cosine = CosineAnnealingLR(optimizer, self.CFG.cosine_epochs, eta_min=1e-7)
            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=self.CFG.multiplier,total_epoch=self.CFG.warmup_epochs, after_scheduler=scheduler_cosine)
            scheduler = scheduler_warmup
        return scheduler
    
    def configure_optimizers(self):
        if self.CFG.optimizer == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.CFG.lr, weight_decay=self.CFG.weight_decay, amsgrad=False)
        elif self.CFG.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.CFG.lr, weight_decay=self.CFG.weight_decay)

        scheduler = self.get_scheduler(optimizer)  
        
        ###https://bleepcoder.com/pytorch-lightning/679052833/how-to-use-reducelronplateau-methon-in-matster-branch
        if self.CFG.scheduler=='ReduceLROnPlateau':
            scheduler = {
                'scheduler': scheduler,
                'reduce_on_plateau': True,
                # val_checkpoint_on is val_loss passed in as checkpoint_on
                'monitor': 'val_loss'
            } 
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
#         images, targets = batch
        
        images, truth_masks, targets = batch
        # 40 = self.CFG.image_size // 16
        truth_masks = F.interpolate(truth_masks, size=(self.CFG.image_size // 16,self.CFG.image_size // 16), mode='bilinear', align_corners=False) 
       
        ### plot images
        if batch_idx < 3 and self.current_epoch == 0:
            if os.path.exists(CFG.save_dir):
                save_img(images, CFG.save_dir + f'/train_{self.current_epoch}_{batch_idx}.png', CFG.save_row_num)
                save_img(truth_masks, CFG.save_dir + f'/mask_{self.current_epoch}_{batch_idx}.png', CFG.save_row_num)
                             
        logits, masks = self(images)   
#         loss = self.train_criterion(logits, targets)
        
        if CFG.criterion == 'BinaryCrossEntropy' or CFG.criterion == 'LabelSmoothingBinaryCrossEntropy' or CFG.criterion == 'BiTemperedLogisticLoss':
            if len(targets.shape) < len(logits.shape):
                targets_onehot = torch.zeros_like(logits)
                targets_onehot.scatter_(1, targets[...,None], 1)
            else:
                targets_onehot = targets
            loss0 = self.train_criterion(logits, targets_onehot)
        else:
            loss0 = self.train_criterion(logits, targets)
        if CFG.seg_type == 'lovasz':
#             loss1 = lovasz_hinge(masks,truth_masks)
            seg_loss = lovasz_hinge(masks,truth_masks)
#             bce_loss = F.binary_cross_entropy_with_logits(masks,truth_masks)
            bce_loss = binary_xloss(masks,truth_masks)
            loss1 = CFG.seg_prob*seg_loss + (1-CFG.seg_prob)*bce_loss
        else: 
            loss1 = F.binary_cross_entropy_with_logits(masks, truth_masks)
        loss = loss0 + loss1
        
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss0', loss0, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss1', loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        logger_logs = {'loss': loss, 'lr': lr}
        output = {
            'loss':loss,
            'progress_bar': logger_logs,
            'log':logger_logs
        }
        return output

    def validation_step(self, batch, batch_idx):
#         images, targets = batch
        images, masks, targets = batch
        
        ### plot images
        if batch_idx < 3 and self.current_epoch == 0:
            if os.path.exists(CFG.save_dir):
                save_img(images, CFG.save_dir + f'/valid_{self.current_epoch}_{batch_idx}.png', CFG.save_row_num)
                save_img(masks, CFG.save_dir + f'/valid_mask_{self.current_epoch}_{batch_idx}.png', CFG.save_row_num)
        
        logits, masks = self(images)
        probability = F.softmax(logits,-1)
        
        val_loss = self.np_loss_cross_entropy(probability.detach().cpu().numpy(), targets.detach().cpu().numpy())

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

         ### gpu
        output = {
            'val_loss': val_loss, 
            'preds': probability, 
            'targets': targets,
        }

        ### cpu
#         output = {
#             'val_loss': val_loss, 
#             'preds': probability.data.cpu().numpy(), 
#             'targets': targets.data.cpu().numpy(),
#         }

        return output

    def np_loss_cross_entropy(self, probability, truth):
        batch_size = len(probability)
        truth = truth.reshape(-1)
        p = probability[np.arange(batch_size),truth]
        loss = -np.log(np.clip(p,1e-6,1))
        loss = loss.mean()
        return loss
    
    def np_metric_roc_auc_by_class(self, probability, truth):
        num_sample, num_label = probability.shape
        score = []
        for i in range(num_label):
            s = roc_auc_score(truth==i, probability[:,i])
            score.append(s)
        score = np.array(score)
        return score

    def np_metric_map_curve_by_class(self, probability, truth):
        num_sample, num_label = probability.shape
        score = []
        for i in range(num_label):
            s = average_precision_score(truth==i, probability[:,i])
            score.append(s)
        score = np.array(score)
        return score
    
    def get_tensor_and_concat(self,tensor):
        gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, tensor)
        # out = torch.cat(gather_t).detach().cpu()
        out = torch.cat(gather_t)
        # del gather_t
        # _ = gc.collect()
        return out
    
    def get_list_and_concat(self,list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
        gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, tensor)
        return torch.cat(gather_t)
    
    def validation_epoch_end(self, outputs):
        ### gpu
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        
        all_preds = self.get_tensor_and_concat(preds)
        all_targets = self.get_tensor_and_concat(targets)
        all_preds = all_preds.data.cpu().numpy()
        
        all_preds = np.nan_to_num(all_preds)
        
        all_targets = all_targets.data.cpu().numpy()
        
        predicts = all_preds.argsort(-1)[:, ::-1]
        val_end_loss = self.np_loss_cross_entropy(all_preds,all_targets)
        
        topk = (predicts== all_targets.reshape(-1,1))
        acc = topk[:, 0]
        topk = topk.mean(0).cumsum()
        acc = [acc[all_targets==i].mean() for i in range(self.CFG.num_classes)]
        
        mAP = self.np_metric_map_curve_by_class(all_preds, all_targets)*(4/6)
        mAP = mAP.mean()
        
#         ### cpu
#         preds = np.concatenate([x['preds'] for x in outputs])
#         targets = np.concatenate([x['targets'] for x in outputs])
        
#         predicts = preds.argsort(-1)[:, ::-1]
#         val_end_loss = self.np_loss_cross_entropy(preds,targets)

#         topk = (predicts==targets.reshape(-1,1))
#         acc  = topk[:, 0]
#         topk = topk.mean(0).cumsum()
#         acc = [acc[targets==i].mean() for i in range(self.CFG.num_classes)]
        
#         mAP  = self.np_metric_map_curve_by_class(preds, targets)*(4/6)
#         mAP = mAP.mean()

        self.log('mAP', mAP)
        LOGGER.info(f'Epoch = {self.current_epoch}, val_end_loss = {val_end_loss:06f}, mAP = {mAP:06f}, top0 = {topk[0]:06f}, top1 = {topk[1]:06f} ')
        logger_logs = {'val_end_loss': val_end_loss, 'mAP': mAP}
        
        output = {
            'mAP':mAP,
            'val_end_loss':val_end_loss,
            'progress_bar': logger_logs,
            'log':logger_logs
        }
        
        return output

    
    
class SIIMDataModule(pl.LightningDataModule):
    def __init__(self, CFG, fold = 0):
        super().__init__()
        self.CFG = CFG
        self.fold = fold
        
    def setup(self, stage=None):
        # In multi-GPU training, this method is run on each GPU. 
        # So ideal for each training/valid split

        df_fold = pd.read_csv(self.CFG.study_folds_csv_path)
        ext_fold = pd.read_csv(self.CFG.external_study_folds_csv_path)
        pseudo_fold = pd.read_csv(self.CFG.pseudo_folds_csv_path)
        if self.CFG.debug:
#         # CFG.epochs = 1
            df_fold = df_fold.sample(n=self.CFG.batch_size*10, random_state=self.CFG.seed).reset_index(drop=True)
        print(df_fold.groupby(['fold']).size())
        
        #---
        df = df_fold.copy()

        #---
        df_train = df[df.fold != self.fold].reset_index(drop=True)
        df_valid = df[df.fold == self.fold].reset_index(drop=True)
        
        ### external data
#         ext_df_train = ext_fold[ext_fold.fold != self.fold].reset_index(drop=True)
#         ext_df_valid = ext_fold[ext_fold.fold == self.fold].reset_index(drop=True)
        
        ### pseudo data
#         pseudo_df_train = pseudo_fold[pseudo_fold.fold != self.fold].reset_index(drop=True)
#         pseudo_df_valid = pseudo_fold[pseudo_fold.fold == self.fold].reset_index(drop=True)
        
#         train_dataset = SIIMMaskDataset(
#             CFG, 
#             df_train, 
#             transforms=get_train_transforms(self.CFG),
#             preprocessing=get_preprocessing())
        
        
        self.valid_dataset = SIIMMaskDataset(
            CFG, 
            df_valid, 
            transforms=get_val_transforms(self.CFG),
            preprocessing=get_preprocessing()) 
        
        if self.CFG.ext:
            external_train_dataset = SIIMMaskExtDataset(
                CFG, 
                ext_fold, 
                transforms=get_train_transforms(self.CFG),
                preprocessing=get_preprocessing())
#             train_dataset = ConcatDataset([train_dataset,external_train_dataset])
            train_dataset = external_train_dataset


        if self.CFG.pseudo:
            pseudo_train_dataset = SIIMMaskPseudoDataset(
                CFG, 
                pseudo_fold, 
                transforms=get_train_transforms(self.CFG),
                preprocessing=get_preprocessing())
            train_dataset = ConcatDataset([train_dataset,pseudo_train_dataset])
#             train_dataset = pseudo_train_dataset

        
        self.train_dataset = train_dataset
        print(f'train dataset len is {len(self.train_dataset)}')
        print(f'valid dataset len is {len(self.valid_dataset)}')
        
     
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.CFG.batch_size, num_workers=4, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.CFG.batch_size*2, num_workers=4, shuffle=False) 
    
def train_loop(CFG, fold, LOGGER):
    LOGGER.info(f'=============== fold: {fold} training =============')
    LOGGER.info(f'Training model {CFG.model_name}, params with image_size={CFG.image_size}, scheduler={CFG.scheduler}, init_lr={CFG.lr}.')
    
    pl.seed_everything(seed=CFG.seed)

    ### load data module
    dm = SIIMDataModule(CFG, fold)
    
    ### init model
    model = SIIMMaskNet(CFG.model_name, CFG.num_classes, pretrained=True)
    

    ### init 
    criterion = create_criterion(CFG, LOGGER)
    
    ### init pl model
    pl_model = SIIMPLModel(CFG, model, criterion)
    
    
    # Folder hack
    tb_logger = TensorBoardLogger(save_dir=CFG.save_dir, name=f'{CFG.model_name}', version=f'fold_{fold}')
    os.makedirs(f'{CFG.save_dir}/{CFG.model_name}', exist_ok=True)
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=tb_logger.log_dir, 
#         filename=f'checkpoint_best',
#         monitor='mAP', 
#         mode='max')
    checkpoint_callback = ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename='{epoch:02d}_{mAP:.6f}',
        save_top_k=5, 
        verbose=True,
        monitor='mAP', 
        mode='max'
        )

#     early_stop_callback = EarlyStopping(
#         monitor='mAP',
#         min_delta=0.0,
#         patience=5,
#         verbose=False,
#         mode='max',
#     )
    
    trainer = pl.Trainer(
        gpus=CFG.gpus,
        precision=CFG.precision,
        max_epochs=CFG.epochs,
#         num_sanity_val_steps=1,
#         resume_from_checkpoint=models_path[fold],
#         num_sanity_val_steps=1 if CFG.debug else 0,
#         checkpoint_callback=checkpoint_callback,#### pl==1.2.4
#         val_check_interval=5.0, # check validation 1 time per 5 epochs
        callbacks=[checkpoint_callback], #### pl==1.3.3
#         check_val_every_n_epoch = 1,
#         accelerator='ddp',
        accumulate_grad_batches=1,
#         gradient_clip_val=1000.0,
        distributed_backend='ddp',
        # amp_backend='native', # or apex
        # amp_level='02', # 01,02, etc...
        # benchmark=True,
        # deterministic=True,
        sync_batchnorm=True,
        logger=tb_logger,
    )
    
    trainer.fit(pl_model, dm)
    
    
if __name__ == '__main__':
    CFG = Config
    if not os.path.exists(CFG.save_dir):
        os.makedirs(CFG.save_dir)   
    LOGGER = get_log(file_name=CFG.save_dir + 'train.log')

    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            train_loop(CFG, fold, LOGGER)
