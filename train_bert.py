import sys
import json
import os
import numpy as np
from data import BERTImageSegmentationDataset, ImageSegmentationDataset
from torch.utils.data import DataLoader
from model.baseline import ImgSegRefExpModel
from model.bert_exp import BertImgSeg
import config
from torch.optim import SGD
import torch
from torch import nn
import time
from data import collate_fn

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

    for batchId, (image_sizes, processed_ims, ground_truth_masks, text_ids, text_mask, text_starts) in enumerate(train_loader):
        optimizer.zero_grad()

        text_ids = text_ids.to(config.device)
        text_mask = text_mask.to(config.device)
        text_starts = text_starts.to(config.device)

        output_mask = model((processed_ims.to(device), (text_ids, text_mask, text_starts)))
    
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

def main():
    print("Starting training")
    # Load model and weights
    # model = ImgSegRefExpModel(mlp_hidden=500, vocab_size=8803, emb_size=1000, lstm_hidden_size=1000)
    model = BertImgSeg()

    # if config.pretrained_wts:
        # pre_trained = torch.load(config.pretrained_model_file)
        # model.load_state_dict(pre_trained)

    model.to(config.device)
    print(model)

    # Combine weight decay regularisation with optimiser
    bert_params = list(model.text_features.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer = torch.optim.Adam(
    [
        {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {"params": model.img_features.res_fc7_full_conv.parameters(), "lr": 3e-4},
        {"params": model.img_features.res_fc8_full_conv.parameters(), "lr": 3e-4},
        {"params": model.mlp1.parameters(), "lr": 3e-4},
        {"params": model.mlp2.parameters(), "lr": 3e-4},
        {"params": model.deconv.parameters(), "lr": 3e-4},
    ],
    lr=5e-5,
    )
    #optimizer = torch.optim.SGD(model.parameters(),lr=config.start_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    train_dataset = BERTImageSegmentationDataset(config.query_file, config.image_dir, config.mask_dir)
    val_dataset = BERTImageSegmentationDataset(config.query_file_val, config.image_dir, config.mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,batch_size=1, shuffle=False, collate_fn=collate_fn)

    model_parameters = filter(lambda p: p.requires_grad, model.img_features.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total params:", params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total params:", params)

    for i in range(0, config.n_epochs):
        print("Training for epoch %d" % (i))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.device)
        print('='*20)

if __name__ == '__main__':
    main()