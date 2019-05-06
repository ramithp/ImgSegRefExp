import sys
import json
import os
import numpy as np
from data import ImageSegmentationDataset, resize_recrop_torch, resize_and_crop
from torch.utils.data import DataLoader
from model.baseline import ImgSegRefExpModel
from model.resnet_exp import ResImgSeg
import config
import torch
from torch import nn
import time
from collections import defaultdict


def compute_mask_IU(masks, target):
    # print (np.sum(np.logical_and(masks, target)))
    assert (target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def calc_length_wise_IoUs(sent_len_I, sent_len_U):
    out = {}
    for key in sent_len_U:
        if sent_len_U[key] != 0:
            out[key] = sent_len_I[key] / sent_len_U[key]
        else:
            print("WARNING: division by zero")
            out[key] = 0
    return out


def test_model(model, test_loader, device):
    start_time = time.time()
    print("Testing\nNumber of batches: {}\tBatch Size: {}\tDataset size: {}".format(len(test_loader),
                                                                                    test_loader.batch_size,
                                                                                    len(test_loader.dataset)))

    sum_I = 0
    sum_U = 0
    model.eval()
    img_num = 0
    sent_len_I = defaultdict(lambda: 0)
    sent_len_U = defaultdict(lambda: 0)

    for batch_idx, (image_sizes, processed_ims, processed_masks, texts) in enumerate(test_loader):
        # print(texts)
        IoU = 0
        batch_time = time.time()
        with torch.no_grad():
            texts = texts.long()
            model_in = time.time()
            output_masks = model((processed_ims.to(device), texts.to(device)))
            text_lens = torch.sum(texts > 0, dim=1)

        # output mask is bsz x 1 x 512 x 512
        output_masks = output_masks.squeeze(1)

        hs, ws = image_sizes

        resize_time = time.time()
        idx = 0
        for (pred_mask, label_mask, h, w) in zip(output_masks, processed_masks, hs, ws):
            # pred = resize_recrop_torch(pred_mask.cpu().detach() > 0, h.item(), w.item())
            # mask = resize_recrop_torch(label_mask.squeeze(0).cpu().detach() > 0, h.item(), w.item())
            pred = resize_and_crop(pred_mask.cpu().detach().numpy() > 0, h.item(), w.item()).astype(np.bool)
            mask = resize_and_crop(label_mask.cpu().detach().numpy().squeeze(0) > 0, h.item(), w.item()).astype(np.bool)
            text_len = text_lens[idx].item()
            #             pdb.set_trace()
            I, U = compute_mask_IU(pred, mask)
            # print (I, U)
            I = float(I)
            U = float(U)

            sum_I += I
            sum_U += U
            sent_len_I[text_len] += I
            sent_len_U[text_len] += U

            if U == 0:
                print("Mask sum", mask.sum())
                continue

            IoU = I / U
            print("For image", img_num, "Img I,U:", I, U, " Image IOU:", IoU)
            img_num += 1

        print("Batch IOU:", sum_I, sum_U, sum_I / sum_U, calc_length_wise_IoUs(sent_len_I, sent_len_U))
    print("Total IoU:", sum_I, sum_U, sum_I / sum_U)
    print("Length wise IOU:")
    for key, value in sorted(calc_length_wise_IoUs(sent_len_I, sent_len_U).items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value))
    print("Completed in :", time.time() - start_time)


def main():
    print("Evaluating IoU")
    # Load model and weights
    # model = ImgSegRefExpModel(mlp_hidden=500, vocab_size=8803, emb_size=1000, lstm_hidden_size=1000)
    model = ResImgSeg(mlp_hidden=500, vocab_size=8803, emb_size=1000, lstm_hidden_size=1000)

    pre_trained = torch.load("project_models_model_dict_ep_7_iter_1000.pt")
    model.load_state_dict(pre_trained)

    model.to(config.device)
    print(model)

    # Combine weight decay regularisation with optimiser
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(int(config.pos_loss_mult),int(config.neg_loss_mult)).to(config.device))

    test_dataset = ImageSegmentationDataset(config.custom_test_set, config.image_dir, config.mask_dir)
    # test_dataset = ImageSegmentationDataset(config.root + 'data/referit_query_test.json', config.image_dir, config.mask_dir)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    test_loss = test_model(model, test_loader, config.device)


if __name__ == '__main__':
    main()
