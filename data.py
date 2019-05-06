from __future__ import absolute_import, division, print_function

import re
import sys
import os
import skimage.io
import numpy as np
import json
import timeit
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import skimage.transform
import numpy as np
import config
from pytorch_pretrained_bert import BertTokenizer

'''
Borrows heavily from
https://github.com/ronghanghu/text_objseg/tree/tensorflow-1.x-compatibility/util
'''
############################## TEXT UTILS ############################## 
UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
PAD_IDENTIFIER = '<pad>'
EOS_IDENTIFIER = '<eos>'

def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]:n for n in range(len(words))}
    return vocab_dict

def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words]
    return vocab_indices

def preprocess_sentence(sentence, vocab_dict, T):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    # # Append '<eos>' symbol to the end
    # vocab_indices.append(vocab_dict[EOS_IDENTIFIER])
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices


############################## IMAGE UTILS ############################## 

def resize_pad_mask(mask, input_h, input_w):
    im_h, im_w = mask.size()
    
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))

    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    transform = transforms.Compose([transforms.ToPILImage(),
                      transforms.Resize([resized_w, resized_h], interpolation=1),
                      transforms.Pad((pad_h, pad_w), fill=0, padding_mode='constant'),
                      transforms.Resize([input_w, input_h], interpolation=1),
                      transforms.ToTensor()])
    img = transform(mask)
    return img

def resize_pad_torch(img, input_h, input_w):
    im_h, im_w = img.size
    
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))

    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    transform = transforms.Compose([transforms.Resize([resized_w, resized_h], interpolation=1),
                  transforms.Pad((pad_h, pad_w), fill=0, padding_mode='constant'),
                  transforms.Resize([input_w, input_h], interpolation=1),
                  transforms.ToTensor()])
    img = transform(img)
    return img
  

def resize_recrop_torch(img, input_h, input_w):
    im_h, im_w = img.shape
    
    # Resize and crop im to input_h x input_w size
    scale = max(input_h / im_h, input_w / im_w)
    
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    
    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    transform = transforms.Compose([transforms.ToPILImage(),
                              transforms.Resize([resized_w, resized_h], interpolation=1),
                              transforms.CenterCrop((resized_w - 2*crop_w, resized_h - 2*crop_h)),
                              transforms.ToTensor()])
    img = transform(img)  
    return img

def resize_and_pad(im, input_h, input_w):
    # Resize and pad im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w, ...] = resized_im

    return new_im

def resize_and_crop(im, input_h, input_w):
    # Resize and crop im to input_h x input_w size
    im_h, im_w = im.shape[:2]

    scale = max(input_h / im_h, input_w / im_w)
    
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))

    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])

    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[...] = resized_im[crop_h:crop_h+input_h, crop_w:crop_w+input_w, ...]

    return new_im

def crop_bboxes_subtract_mean(im, bboxes, crop_size, image_mean):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    im = skimage.img_as_ubyte(im)
    num_bbox = bboxes.shape[0]
    imcrop_batch = np.zeros((num_bbox, crop_size, crop_size, 3), dtype=np.float32)
    for n_bbox in range(bboxes.shape[0]):
        xmin, ymin, xmax, ymax = bboxes[n_bbox]
        # crop and resize
        imcrop = im[ymin:ymax+1, xmin:xmax+1, :]
        imcrop_batch[n_bbox, ...] = skimage.img_as_ubyte(
            skimage.transform.resize(imcrop, [crop_size, crop_size]))
    imcrop_batch -= image_mean
    return imcrop_batch

def bboxes_from_masks(masks):
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    num_mask = masks.shape[0]
    bboxes = np.zeros((num_mask, 4), dtype=np.int32)
    for n_mask in range(num_mask):
        idx = np.nonzero(masks[n_mask])
        xmin, xmax = np.min(idx[1]), np.max(idx[1])
        ymin, ymax = np.min(idx[0]), np.max(idx[0])
        bboxes[n_mask, :] = [xmin, ymin, xmax, ymax]
    return bboxes

def crop_masks_subtract_mean(im, masks, crop_size, image_mean):
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    num_mask = masks.shape[0]

    im = skimage.img_as_ubyte(im)
    bboxes = bboxes_from_masks(masks)
    imcrop_batch = np.zeros((num_mask, crop_size, crop_size, 3), dtype=np.float32)
    for n_mask in range(num_mask):
        xmin, ymin, xmax, ymax = bboxes[n_mask]

        # crop and resize
        im_masked = im.copy()
        mask = masks[n_mask, ..., np.newaxis]
        im_masked *= mask
        im_masked += image_mean.astype(np.uint8) * (1 - mask)
        imcrop = im_masked[ymin:ymax+1, xmin:xmax+1, :]
        imcrop_batch[n_mask, ...] = skimage.img_as_ubyte(skimage.transform.resize(imcrop, [224, 224]))

    imcrop_batch -= image_mean
    return imcrop_batch

############################## BERT UTILS ############################## 

# Borrowed from https://github.com/bheinzerling/dougu/blob/2f54b14d588f17d77b7a8bca9f4e5eb38d6a2805/dougu/bert.py#L98
# https://github.com/bheinzerling/dougu
def subword_tokenize(tokenizer, tokens, cls='[CLS]', sep='[SEP]'):
        flatten = lambda l: [item for sublist in l for item in sublist]
        subwords = list(map(tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [cls] + list(flatten(subwords)) + [sep]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])

        # print(subwords, token_start_idxs)
        
        return subwords, token_start_idxs

def convert_tokens_to_ids(tokenizer, tokens, max_len, pad=True):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.LongTensor([token_ids]).to(device=config.device)
        if pad:
            padded_ids = torch.LongTensor(1, max_len).zero_().to(config.device)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.LongTensor(1, max_len).zero_().to(config.device)
            mask[0, :ids.size(1)] = 1
            # print(padded_ids, mask)
            return padded_ids, mask
        else:
            return ids

def subword_tokenize_to_ids(tokenizer, tokens, max_len):
    subwords, token_start_idxs = subword_tokenize(tokenizer, tokens)
    subword_ids, mask = convert_tokens_to_ids(tokenizer, subwords, max_len)
    token_starts = torch.zeros(1, max_len).to(subword_ids)
    # token_starts[0, token_start_idxs] = 1
    return subword_ids, mask, token_starts


############################## DATASET CLASS(ES) ############################## 

class ImageSegmentationDataset(Dataset):
    @staticmethod
    def load_mask(mask_path):
        mat = sio.loadmat(mask_path)
        mask = (mat['segimg_t'] == 0)
        return mask

    def __init__(self, query_file, root_directory_image, root_directory_mask, test = False, transform = None):
        
        self.query_dict = json.load(open(config.query_file))
        self.bbox_dict = json.load(open(config.bbox_file))
        self.imcrop_dict = json.load(open(config.imcrop_file))
        self.imsize_dict = json.load(open(config.imsize_file))
        self.imlist = list({name.split('_', 1)[0] + '.jpg' for name in self.query_dict})
        self.vocab_dict = load_vocab_dict_from_file(config.vocab_file)

        self.flat_query_dict = {imname: [] for imname in self.imlist}
        for imname in self.imlist:
            this_imcrop_names = self.imcrop_dict[imname]
            for imcrop_name in this_imcrop_names:
                gt_bbox = self.bbox_dict[imcrop_name]
                if imcrop_name not in self.query_dict:
                    continue
                this_descriptions = self.query_dict[imcrop_name]
                for description in this_descriptions:
                    self.flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

        self.root_directory_image = root_directory_image
        self.root_directory_mask = root_directory_mask
        
        self.query_file = query_file
        self.query_dict = json.load(open(self.query_file))
        self.query_keys = []
        for key in self.query_dict:
            value_list = self.query_dict[key]
            for idx,_ in enumerate(value_list):
                self.query_keys.append("{}_{}".format(key,idx))
        self.dataset_length = len(self.query_keys)
        self.test_flag = test

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        current_img_name = self.query_keys[idx]
        current_img_main = int(current_img_name.split("_")[0])
        current_img_crop = int(current_img_name.split("_")[1])
        current_img_text = int(current_img_name.split("_")[2])
        
        # Get image to pass around
        img_name = os.path.join(config.image_dir + str(current_img_main) + '.jpg')
        img = Image.open(img_name)
        img = img.convert('RGB')

        processed_im = resize_pad_torch(img, config.input_H, config.input_W)
        
        # Get text
        img_text = self.query_dict["{}_{}".format(current_img_main, current_img_crop)][current_img_text]
        text_seq_val = np.zeros((config.T, config.N), dtype=np.float32)
        text_seq_val[:, 0] = preprocess_sentence(img_text, self.vocab_dict, config.T)

        if len(processed_im.size()) == 2:
            print("HALLO")
            processed_im = np.tile(processed_im[:, :, np.newaxis], (1, 1, 3))

        if (not self.test_flag):
            mask_file_name = os.path.join(self.root_directory_mask,"{}_{}.mat".format(current_img_main, current_img_crop))
            mask = torch.Tensor(ImageSegmentationDataset.load_mask(mask_file_name).astype(np.float32))
            processed_mask = resize_pad_mask(mask, config.input_H, config.input_W)
            return img.size, processed_im, processed_mask, text_seq_val[:, 0] 
                    
        return processed_im, text_seq_val[:, 0]



class BERTImageSegmentationDataset(Dataset):
    @staticmethod
    def load_mask(mask_path):
        mat = sio.loadmat(mask_path)
        mask = (mat['segimg_t'] == 0)
        return mask

    def __init__(self, query_file, root_directory_image, root_directory_mask, test = False, transform = None):
        
        self.query_dict = json.load(open(config.query_file))
        self.bbox_dict = json.load(open(config.bbox_file))
        self.imcrop_dict = json.load(open(config.imcrop_file))
        self.imsize_dict = json.load(open(config.imsize_file))
        self.imlist = list({name.split('_', 1)[0] + '.jpg' for name in self.query_dict})
        self.vocab_dict = load_vocab_dict_from_file(config.vocab_file)

        self.flat_query_dict = {imname: [] for imname in self.imlist}
        for imname in self.imlist:
            this_imcrop_names = self.imcrop_dict[imname]
            for imcrop_name in this_imcrop_names:
                gt_bbox = self.bbox_dict[imcrop_name]
                if imcrop_name not in self.query_dict:
                    continue
                this_descriptions = self.query_dict[imcrop_name]
                for description in this_descriptions:
                    self.flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

        self.root_directory_image = root_directory_image
        self.root_directory_mask = root_directory_mask
        
        self.query_file = query_file
        self.query_dict = json.load(open(self.query_file))
        self.query_keys = []
        for key in self.query_dict:
            value_list = self.query_dict[key]
            for idx,_ in enumerate(value_list):
                self.query_keys.append("{}_{}".format(key,idx))
        self.dataset_length = len(self.query_keys)
        self.test_flag = test
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        current_img_name = self.query_keys[idx]
        current_img_main = int(current_img_name.split("_")[0])
        current_img_crop = int(current_img_name.split("_")[1])
        current_img_text = int(current_img_name.split("_")[2])
        
        # Get image to pass around
        img_name = os.path.join(config.image_dir + str(current_img_main) + '.jpg')
        img = Image.open(img_name)
        img = img.convert('RGB')

        processed_im = resize_pad_torch(img, config.input_H, config.input_W)
        
        # Get text
        img_text = self.query_dict["{}_{}".format(current_img_main, current_img_crop)][current_img_text]
        text_seq_val = np.zeros((config.T, config.N), dtype=np.float32)
        text_seq_val[:, 0] = preprocess_sentence(img_text, self.vocab_dict, config.T)

        features = {}
        split = subword_tokenize_to_ids(self.tokenizer, img_text.split(), config.T)
        features["bert_ids"], features["bert_mask"], features["bert_token_starts"] = split

        if (not self.test_flag):
            mask_file_name = os.path.join(self.root_directory_mask,"{}_{}.mat".format(current_img_main, current_img_crop))
            mask = torch.Tensor(ImageSegmentationDataset.load_mask(mask_file_name).astype(np.float32))
            processed_mask = resize_pad_mask(mask, config.input_H, config.input_W)
            return (img.size, processed_im, processed_mask, features) 
                    
        return processed_im, text_seq_val[:, 0]

def collate_fn(featurized_batch):
    image_sizes = [feat[0] for feat in featurized_batch]
    processed_ims = torch.stack([feat[1] for feat in featurized_batch])
    ground_truth_masks = torch.stack([feat[2] for feat in featurized_batch])
    text_ids = torch.stack([features[3]["bert_ids"] for features in featurized_batch])
    text_mask = torch.stack([features[3]["bert_mask"] for features in featurized_batch])
    text_starts = torch.stack([features[3]["bert_token_starts"] for features in featurized_batch])

    return (image_sizes, processed_ims, ground_truth_masks, text_ids.squeeze(1), text_mask.squeeze(1), text_starts.squeeze(1))