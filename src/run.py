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
    kernel_type = '120_myrainnet_rsp_weight'
    
    original_width = 120
    original_height = 120
    
#     resize_width = 128
#     resize_height = 128
    
    # Set Model-related params
    seed = 0
    num_workers = 6
    num_folds = 5
    
    batch_size = 64
    init_lr =  3e-4
    num_epochs = 100
    start_from_epoch = 1
    stop_at_epoch = 999
    
    use_amp = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(config.seed)

def train_epoch(CFG, loader, optimizer):
    model.train()
    train_losses = [ ]
    train_losses_rsp = []
    train_losses_torch = []
    bar = tqdm(loader)
    for batch in bar:
        img, targets = batch[0], batch[1]
        img = img.float().to(CFG.device)
        targets = targets.float().to(CFG.device)
        
        loss_func = nn.L1Loss()
        
        optimizer.zero_grad()
        
        logits = model(img)

        loss = loss_func( logits , targets )
        for idx in [[67, 28], [64, 46], [69, 57], [66, 62], [64, 68], [71, 70], [87, 56]]:
            train_losses_rsp += [loss_func(logits[:, :, idx[0], idx[1]]/2, targets[:, :, idx[0], idx[1]]/2).detach().cpu().numpy()]
            # train_losses_torch += [loss_func(logits[:, :, idx[0], idx[1]], targets[:, :, idx[0], idx[1]])]
        # else:
        #     loss += torch.mean(torch.Tensor(train_losses_torch))

        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_losses.append( loss_np )
        # smooth_loss = sum(train_losses[-20:]) / min(len(train_losses), 20)
        bar.set_description('loss: %.5f' % (loss_np))
    else:
        train_losses = np.mean(train_losses)
        train_losses_rsp = np.mean(train_losses_rsp)
        
    return train_losses, train_losses_rsp



def val_epoch(CFG, loader, get_output=False):
    
    model.eval()
    
    val_losses = [ ]
    valid_losses_rsp = []
    
    bar = tqdm(loader)
    
    with torch.no_grad():
        for batch in bar: 
            img, targets = batch[0], batch[1]
            img = img.float().to(CFG.device)
            targets = targets.float().to(CFG.device)
        
            logits = model(img)
            
            loss_func = nn.L1Loss()
            loss =  loss_func(logits , targets )
            for idx in [[67, 28], [64, 46], [69, 57], [66, 62], [64, 68], [71, 70], [87, 56]]:
                valid_losses_rsp += [loss_func(logits[:, :, idx[0], idx[1]]/2, targets[:, :, idx[0], idx[1]]/2).detach().cpu().numpy()]
                                         
            val_losses.append( loss.detach().cpu().numpy() )
            
        val_losses = np.mean(val_losses)
        valid_losses_rsp = np.mean(valid_losses_rsp)
        
    return val_losses, valid_losses_rsp

data_path = [os.path.join(config.path, i) for i in os.listdir(config.path)]
data_path = np.array(sorted(data_path))#[:1000]

# data_path = [os.path.join(config.path, i) for i in os.listdir(config.path) if 'modify' in i and '3.npy' in i]
# data_path += ['../data/train_split/train_modify_00000_0.npy', '../data/train_split/train_modify_00000_1.npy', '../data/train_split/train_modify_00000_2.npy']
# data_path = np.array(sorted(data_path))
# test_img_path = "../input/data/test/"

kf = KFold(n_splits=config.num_folds, random_state=0, shuffle=True)
n_splits = [[trn_idx, val_idx] for trn_idx, val_idx in kf.split(data_path)]
fold_ind=0

train_path = data_path[n_splits[fold_ind][0]]
valid_path = data_path[n_splits[fold_ind][1]]
# valid_path = [f.replace('train_modify', 'train') for f in data_path[n_splits[fold_ind][1]]]

train_img_path = [f.replace('train/', 'train_modify/') for f in train_path]
valid_img_path = [f.replace('train/', 'train_modify/') for f in valid_path]

# train_dataset = RainDataset(config, train_img_path, train_path, mode='train')
# valid_dataset = RainDataset(config, valid_img_path, valid_path, mode='valid')

train_dataset = RainDataset(config, train_path, train_path, mode='train')
valid_dataset = RainDataset(config, valid_path, valid_path, mode='valid')

train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size, 
                            num_workers=config.num_workers, 
                            pin_memory=True, 
                            shuffle=True)
valid_loader = DataLoader(valid_dataset,
                            batch_size=config.batch_size , 
                            num_workers=config.num_workers, 
                            pin_memory=False, 
                            shuffle=True)


# model
model = my_rainnet(config).to(config.device)
# model.load_state_dict(torch.load('model/120_myrainnet_rsp_fold0.pth')['model_state_dict'])
# model = UNet(in_channel=4,out_channel=1).to(config.device)

optimizer = optim.Adam(model.parameters(), lr = config.init_lr)

config.use_amp=False
if config.use_amp:
    model, optimizer = amp.initialize(
        model, optimizer, verbosity=0
    )
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
# lr scheduler
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.num_epochs-1)
scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

# Train & Valid Loop
metric_loss = np.inf
model_file = os.path.join( config.model_dir, f"{config.kernel_type}_fold{fold_ind}.pth" )


print('\n\n\n\n\nTraining....................................')
for epoch in range( config.start_from_epoch, config.num_epochs +1 ):
    print(time.ctime(), "Epoch : ", epoch)
    scheduler_warmup.step(epoch - 1)
    
    criterion = nn.L1Loss()
    
    train_losses, train_losses_rsp = train_epoch(config, train_loader, optimizer)
    valid_losses, valid_losses_rsp  = val_epoch(config, valid_loader)
    
    if valid_losses_rsp < metric_loss :
        content = time.ctime() + ' ' + f'Fold {fold_ind}, Epoch {epoch}, \
            lr: {optimizer.param_groups[0]["lr"]:.7f}, \
            train loss: {train_losses:.5f}, \
            valid loss: {valid_losses:.5f}. \
            train rsp loss: {train_losses_rsp:.5f}, \
            valid rsp loss: {valid_losses_rsp:.5f}.'

        print(content)
        with open( os.path.join( config.log_dir, f"{config.kernel_type}.txt"), "a" )  as appender:
            appender.write( content +'\n' )
        
        print('RSP MAE ({:.6f} --> {:.6f}). Saving model ...'.format(metric_loss, valid_losses_rsp))
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, model_file)            
        metric_loss = valid_losses_rsp
    else:
        content = time.ctime() + ' ' + f'Fold {fold_ind}, Epoch {epoch}, \
            lr: {optimizer.param_groups[0]["lr"]:.7f}, \
            train loss: {train_losses:.5f}, \
            valid loss: {valid_losses:.5f}. \
            train rsp loss: {train_losses_rsp:.5f}, \
            valid rsp loss: {valid_losses_rsp:.5f}.'
        print('Unimproved : ', content)
    
    if epoch == config.stop_at_epoch:
        print(time.ctime(), 'Training Finished!')
        break
        
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(config.model_dir, f'{config.kernel_type_type}_fold{fold_ind}_final.pth'))


# # if __name__ == '__main__':
    
