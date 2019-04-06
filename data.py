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
from utils import text_processing, im_processing

################################################################################
# Parameters
################################################################################

T = 20
N = 1
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Evaluation Param
score_thresh = 1e-9

T = 20
N = 1
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

root = '/Users/shubhammehrotra/text_objseg/exp-referit/'

image_dir = root + 'referit-dataset/images/'
mask_dir = root + 'referit-dataset/mask/'
query_file = root + 'data/referit_query_test.json'
bbox_file = root + 'data/referit_bbox.json'
imcrop_file = root + 'data/referit_imcrop.json'
imsize_file = root + 'data/referit_imsize.json'
vocab_file = root + 'data/vocabulary_referit.txt'


query_dict = json.load(open(query_file))
bbox_dict = json.load(open(bbox_file))
imcrop_dict = json.load(open(imcrop_file))
imsize_dict = json.load(open(imsize_file))
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Flatten the annotations
################################################################################

flat_query_dict = {imname: [] for imname in imlist}
for imname in imlist:
    this_imcrop_names = imcrop_dict[imname]
    for imcrop_name in this_imcrop_names:
        gt_bbox = bbox_dict[imcrop_name]
        if imcrop_name not in query_dict:
            continue
        this_descriptions = query_dict[imcrop_name]
        for description in this_descriptions:
            flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

def load_mask(mask_path):
    mat = sio.loadmat(mask_path)
    mask = (mat['segimg_t'] == 0)
    return mask

class ImageSegmentationDataset(Dataset):

    def __init__(self, query_file, root_directory_image, root_directory_mask, test = False, transform = None):
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
        self.transform = transform

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        current_img_name = self.query_keys[idx]
        current_img_main = int(current_img_name.split("_")[0])
        current_img_crop = int(current_img_name.split("_")[1])
        current_img_text = int(current_img_name.split("_")[2])
        img_text = self.query_dict["{}_{}".format(current_img_main,current_img_crop)][current_img_text]
        image = skimage.io.imread(image_dir + imname)
        processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(image, input_H, input_W))
        if processed_im.ndim == 2:
            processed_im = np.tile(processed_im[:, :, np.newaxis], (1, 1, 3))
        processed_im = transforms.ToTensor()(processed_im)

        if (not self.test_flag):
            mask_file_name = os.path.join(self.root_directory_mask,"{}_{}.mat".format(current_img_main, current_img_crop))
            mask = load_mask(mask_file_name).astype(np.float32)
            processed_mask = im_processing.resize_and_pad(mask, input_H, input_W)
            return processed_im, img_text, processed_mask
        
        return processed_im, img_text


train_dataset = ImageSegmentationDataset(query_file, image_dir, mask_dir)
train_loader = DataLoader(train_dataset, batch_size = 2, shuffle=True)

test_dataset = ImageSegmentationDataset(query_file, image_dir, mask_dir, test = True)
test_loader = DataLoader(test_dataset, batch_size = 200)

for batch_idx, (image, text, mask) in enumerate(train_loader):
    print (batch_idx, image.shape)
    break
