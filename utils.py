from __future__ import absolute_import, division, print_function

import skimage.transform
import numpy as np

import re

def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]:n for n in range(len(words))}
    return vocab_dict

UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words]
    return vocab_indices

PAD_IDENTIFIER = '<pad>'
EOS_IDENTIFIER = '<eos>'
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