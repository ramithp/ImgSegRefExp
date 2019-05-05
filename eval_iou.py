import sys
import json
import os
import numpy as np
from data import ImageSegmentationDataset
from torch.utils.data import DataLoader
from model.baseline import ImgSegRefExpModel
import config
from torch.optim import SGD
import torch
from torch import nn
import time

def eval_model(model, test_loader, criterion, device, epoch_num):
    print("Testing\nNumber of batches: {}\tBatch Size: {}\tDataset size: {}".format(len(test_loader), 
                                                                                        test_loader.batch_size,
                                                                                        len(test_loader.dataset)))

    model.eval()
    for batch_idx, (image_sizes, processed_ims, processed_masks, texts) in enumerate(test_loader):
        IoU = 0
        batch_time = time.time()
        with torch.no_grad():
            texts = texts.long()
            model_in = time.time()
            output_masks = model((processed_ims.to(device), texts.to(device)))

        # output mask is bsz x 1 x 512 x 512
        output_masks = output_masks.squeeze(1)
               
        hs, ws = image_sizes

        resize_time=time.time()
        for (pred_mask, label_mask, h, w) in zip(output_masks, processed_masks, hs, ws):
            pred = resize_recrop_torch(pred_mask.cpu().detach() > 0, h.item(), w.item())
            mask = resize_recrop_torch(label_mask.squeeze(0).cpu().detach() > 0, h.item(), w.item())

    #         pred = resize_and_crop(pred_mask.cpu().detach().numpy() > 0, h.item(), w.item()).astype(np.bool)
    #         mask = resize_and_crop(label_mask.cpu().detach().numpy() > 0, h.item(), w.item()).astype(np.bool)

            I, U = compute_mask_IU(pred, mask)
            #print (I, U)
            I = float(I)
            U = float(U)
            sum_I += I
            sum_U += U
            if U ==0:
                print("Mask sum", mask.sum())
                continue
                IoU += I / U
        print ("IOU:", IoU/32)

def main():
    print("Starting training")
    # Load model and weights
    model = ImgSegRefExpModel(mlp_hidden=500, vocab_size=8803, emb_size=1000, lstm_hidden_size=1000)

    if config.pretrained_wts:
        pre_trained = torch.load(config.pretrained_model_file)
        model.load_state_dict(pre_trained)

    model.to(config.device)
    print(model)

    # Combine weight decay regularisation with optimiser
    optimizer = torch.optim.SGD(model.parameters(),lr=config.start_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(int(config.pos_loss_mult),int(config.neg_loss_mult)).to(config.device))

    train_dataset = ImageSegmentationDataset(config.query_file, config.image_dir, config.mask_dir)
    val_dataset = ImageSegmentationDataset(config.query_file_val, config.image_dir, config.mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=10, shuffle=False)


    for i in range(0, config.n_epochs):
        print("Training for epoch %d" % (i))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device, )
        test_loss = eval_model(model, val_loader, criterion, config.device)

        torch.save(model.state_dict(), config.SAVE_PATH_PREFIX + '/models/model_dict_'+ str(i) + '.pt')

        print('='*20)

if __name__ == '__main__':
    main()