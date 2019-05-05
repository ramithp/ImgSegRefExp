import sys
import json
import os
import numpy as np
from data import ImageSegmentationDataset
from torch.utils.data import DataLoader
from model.baseline import ImgSegRefExpModel
from model.resnet_exp import ResImgSeg
import config
from torch.optim import SGD
import torch
from torch import nn
import time

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch_num, device=config.device):
    print("Training\n Number of batches: {}\tBatch Size: {}\tDataset size: {}".format(len(train_loader),train_loader.batch_size,len(train_loader.dataset)))
    start = time.time()
    model.train()
    cls_loss_avg = 0.0
    end_time = 0
    batch_start = time.time()

    for batchId, (image_sizes, processed_ims, ground_truth_masks, texts) in enumerate(train_loader):
        optimizer.zero_grad()
        texts = texts.long()

        output_mask = model((processed_ims.to(device), texts.to(device)))
        output_mask = output_mask.squeeze(1)        
        ground_truth_masks = ground_truth_masks.to(device).squeeze(1).float()

        # Adapted from https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
        # and https://github.com/ramithp/text_objseg/blob/tensorflow-1.x-compatibility/util/loss.py
        pos_loss_mult = 1.
        neg_loss_mult = 1. /3.
        loss_mult = (ground_truth_masks * (pos_loss_mult-neg_loss_mult)) +  neg_loss_mult

        # Classification loss as the average of weighed per-score loss
        cls_loss = (criterion(output_mask, ground_truth_masks) * loss_mult).mean()

        cls_loss.backward()
        cls_loss_avg = config.decay * cls_loss + (1 - config.decay) * cls_loss.item()
        optimizer.step()
        scheduler.step()
        
        if batchId % 20 == 0:
            print("Batch Time with data loading = {}s, Batch #{}: Loss = {}\tAvg Loss: {}\tTime: {}s".format(time.time() - end_time,
                                                                                                            batchId,
                                                                                                            cls_loss.item(),
                                                                                                            cls_loss_avg,time.time() - batch_start))
        if batchId % 500 == 0:
            torch.save(model.state_dict(), 'model_dict_ep_'+ str(epoch_num) + '_iter_' + str(batchId) + '.pt')

        end_time = time.time()

    print('Batch Loss (avg) = {}, lr = {}, time = {}s'.format(cls_loss_avg, 
                                                             get_lr(optimizer),
                                                             time.time()-start))

def eval_model(model, val_loader, criterion, epoch_num, device):
    print("Validating\nNumber of batches: {}\tBatch Size: {}\tDataset size: {}".format(len(val_loader), 
                                                                                        val_loader.batch_size,
                                                                                        len(val_loader.dataset)))
    cls_loss_avg = 0.0
    model.eval()
    with torch.no_grad():
        batch_start = time.time()
        for batchId, (image_sizes, processed_ims, ground_truth_masks, texts) in enumerate(val_loader):
            texts = texts.long()
            output_masks = model((processed_ims.to(device), texts.to(device)))
            output_masks = output_masks.squeeze(1)
            ground_truth_masks = ground_truth_masks.to(device).squeeze(1).float()

            cls_loss_val = criterion(output_masks, ground_truth_masks)
            cls_loss_avg = config.decay * cls_loss_avg + (1 - config.decay) * cls_loss_val.item()
            
            if batchId % 100 == 0:
                print("Batch #{}: Loss = {}\tAvg Loss: {}\tTime: {}s".format(batchId,
                                                                            cls_loss_val.item(),
                                                                            cls_loss_avg,time.time() - batch_start))
        print('Batch Loss (avg) = {}, lr = {}, time = {}s'.format(cls_loss_avg,
                                                                 get_lr(optimiser),
                                                                 time.time()))

def main():
    print("Starting training")
    # Load model and weights
    # model = ImgSegRefExpModel(mlp_hidden=500, vocab_size=8803, emb_size=1000, lstm_hidden_size=1000)
    model = ResImgSeg(mlp_hidden=500, vocab_size=8803, emb_size=1000, lstm_hidden_size=1000)

    # if config.pretrained_wts:
        # pre_trained = torch.load(config.pretrained_model_file)
        # model.load_state_dict(pre_trained)

    model.to(config.device)
    print(model)

    # Combine weight decay regularisation with optimiser
    optimizer = torch.optim.SGD(model.parameters(),lr=config.start_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    train_dataset = ImageSegmentationDataset(config.query_file, config.image_dir, config.mask_dir)
    val_dataset = ImageSegmentationDataset(config.query_file_val, config.image_dir, config.mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

    model_parameters = filter(lambda p: p.requires_grad, model.img_features.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total params:", params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total params:", params)

    for i in range(0, config.n_epochs):
        print("Training for epoch %d" % (i))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.device)
        test_loss = eval_model(model, val_loader, criterion, config.device)

        print('='*20)

if __name__ == '__main__':
    main()