import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams

import os,random, time,datetime, cv2, albumentations, math, joblib, gc, math
from PIL import Image
from glob import glob

from tqdm import tqdm, trange
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable 
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import RandomSampler
import torch.nn.functional as F
from torch.backends import cudnn

from scheduler import GradualWarmupScheduler
from scheduler import GradualWarmupSchedulerV2

from rainnet import RainNet, SmallRainNet, my_rainnet
import geffnet
# from efficientnet_pytorch import EfficientNet

from typing import Dict, Tuple, Any

import apex
from apex import amp, optimizers

from dataloader import RainDataset
import os
from sklearn.model_selection import KFold

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class config:
    path = '../data/train/'

    model_dir = "model"
    log_dir = "log"
    kernel_type = "120_myrainnet_rsp_weight"
    
    original_width = 120
    original_height = 120
    
#     resize_width = 128
#     resize_height = 128
    
    # Set Model-related params
    seed = 0
    num_workers = 0
    num_folds = 5
    
    batch_size = 16
    init_lr =  1e-3
    num_epochs = 40
    start_from_epoch = 1
    stop_at_epoch = 999
    
    use_amp = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(config.seed)

################################ dataset

data_path = [os.path.join(config.path, i) for i in os.listdir(config.path)]
data_path = np.array(sorted(data_path))#[:1000]

kf = KFold(n_splits=config.num_folds, random_state=0, shuffle=True)
n_splits = [[trn_idx, val_idx] for trn_idx, val_idx in kf.split(data_path)]
fold_ind=0

train_path = data_path[n_splits[fold_ind][0]]
valid_path = data_path[n_splits[fold_ind][1]]
# valid_path = [f.replace('train_modify', 'train') for f in data_path[n_splits[fold_ind][1]]]

train_img_path = [f.replace('train/', 'train_modify/') for f in train_path]
valid_img_path = [f.replace('train/', 'train_modify/') for f in valid_path]

# train_dataset = RainDataset(config, train_img_path, train_path, mode='train')
valid_dataset = RainDataset(config, valid_path, valid_path, mode='valid')

# train_loader = DataLoader(train_dataset, 
#                             batch_size=config.batch_size, 
#                             num_workers=config.num_workers, 
#                             pin_memory=True, 
#                             shuffle=True)
valid_loader = DataLoader(valid_dataset,
                            batch_size=1, 
                            num_workers=config.num_workers, 
                            pin_memory=False, 
                            shuffle=False)

test_path = [os.path.join('../data/test', f) for f in sorted(os.listdir('../data/test'))]

test_dataset = RainDataset(config, test_path, mode='test')
test_loader = DataLoader(test_dataset,
                            batch_size=1, 
                            num_workers=config.num_workers, 
                            pin_memory=False, 
                            shuffle=False)

# model
model = my_rainnet(config).to(config.device)
model_file = os.path.join( config.model_dir, f"{config.kernel_type}_fold{fold_ind}.pth" )
model.load_state_dict(torch.load(model_file)['model_state_dict'])
model.eval()
# optimizer = optim.Adam(model.parameters(), lr = config.init_lr)

config.use_amp=False
if config.use_amp:
    model, optimizer = amp.initialize(
        model, optimizer, verbosity=0
    )
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
# lr scheduler
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.num_epochs-1)
# scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

# Train & Valid Loop
# metric_loss = np.inf
# model_file = os.path.join( config.model_dir, f"{config.kernel_type}_fold{fold_ind}.pth" )

def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

def reverse_get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

preds = []
valid_losses_rsp = []
val_losses = []

n_test=1
print('\nEvaluation....................................')
bar = tqdm(valid_loader)
with torch.no_grad():
    for batch in bar: 
        img, targets = batch[0], batch[1]
        img = img.float().to(config.device)
        targets = targets.float().to(config.device)
        logits = torch.zeros((1, 1, 120, 120)).to(config.device)
        for I in range(n_test):
            l = reverse_get_trans(model(get_trans(img, I)), I)
            logits += l
        else:
            logits /= n_test
        loss_func = nn.L1Loss()

        for idx in [[67, 28], [64, 46], [69, 57], [66, 62], [64, 68], [71, 70], [87, 56]]:
            logits[:, :, idx[0], idx[1]] = logits[:, :, idx[0], idx[1]]/2
            targets[:, :, idx[0], idx[1]] = targets[:, :, idx[0], idx[1]]/2
            
            valid_losses_rsp += [loss_func(logits[:, :, idx[0], idx[1]], targets[:, :, idx[0], idx[1]]).detach().cpu().numpy()]

        loss =  loss_func(logits , targets )
                                        
        val_losses.append( loss.detach().cpu().numpy() )
        preds.append( logits.detach().cpu().numpy() )
        
    val_losses = np.mean(val_losses)
    valid_losses_rsp = np.mean(valid_losses_rsp)
preds = np.array(preds)
print(preds, preds.shape, val_losses, valid_losses_rsp)

np.save('../submit/oof_rsp_weight.npy', preds)

# print('\nEvaluation....................................')
# bar = tqdm(valid_loader)
# with torch.no_grad():
#     for batch in bar: 
#         img, targets = batch[0], batch[1]
#         img = img.float().to(config.device)
#         targets = targets.float().to(config.device)
        
#         logits = model(img)
        
#         loss_func = nn.L1Loss()
#         loss =  loss_func(logits , targets )
                                        
#         val_losses.append( loss.detach().cpu().numpy() )
#         preds.append( logits.detach().cpu().numpy() )
        
#     val_losses = np.mean(val_losses)
# preds = np.array(preds)
# print(preds, preds.shape, val_losses)

# np.save('../submit/oof_120_10folds.npy', preds)

preds = []
bar = tqdm(test_loader)
with torch.no_grad():
    for img in bar: 
        img = img.float().to(config.device)
    
        logits = model(img)
        for idx in [[67, 28], [64, 46], [69, 57], [66, 62], [64, 68], [71, 70], [87, 56]]:
            logits[:,:,idx[0],idx[1]] = logits[:,:,idx[0],idx[1]]/2
        
        preds.append( logits.detach().cpu().numpy() )
        
preds = np.array(preds)
print(preds, preds.shape)

np.save('../submit/folds5_rsp_weight.npy', preds)

