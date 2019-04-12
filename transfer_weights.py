from __future__ import absolute_import, division, print_function

import sys
# import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit
import matplotlib.pyplot as plt

sys.path.append('../')

from models import text_objseg_model as segmodel
from util import text_processing

# trained model
pretrained_model = '../exp-referit/tfmodel/referit_fc8_seg_highres_iter_18000.tfmodel'
vocab_file = '../exp-referit/data/vocabulary_referit.txt'

# Load the model
# Model Param
T = 20
N = 1
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imcrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])

# Outputs
scores = segmodel.text_objseg_upsample32s(text_seq_batch, imcrop_batch, num_vocab,
                                          embed_dim, lstm_dim, mlp_hidden_dims,
                                          vgg_dropout=False, mlp_dropout=False)

# Load pretrained model
# Update variable names for TF 1.0.0 or higher
variable_name_mapping= None
if tf.__version__.split('.')[0] == '1':
    variable_name_mapping = {
        v.op.name.replace(
            'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
            'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix').replace(
            'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
            'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias'): v
        for v in tf.global_variables()}

snapshot_restorer = tf.train.Saver(variable_name_mapping)
sess = tf.Session()
snapshot_restorer.restore(sess, pretrained_model)

# Load vocabulary
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

from collections import OrderedDict
import torch
sv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

weights_dict = OrderedDict()
for s in sv:
    weights_dict[s.name] = torch.from_numpy(sess.run(s))
    
weights = OrderedDict()
weights['features.0.weight'] = weights_dict['vgg_local/conv1_1/weights:0'].permute(3,2,0,1)
weights['features.0.bias'] = weights_dict['vgg_local/conv1_1/biases:0']
weights['features.2.weight'] = weights_dict['vgg_local/conv1_2/weights:0'].permute(3,2,0,1)
weights['features.2.bias'] = weights_dict['vgg_local/conv1_2/biases:0']
weights['features.5.weight'] = weights_dict['vgg_local/conv2_1/weights:0'].permute(3,2,0,1)
weights['features.5.bias'] = weights_dict['vgg_local/conv2_1/biases:0']
weights['features.7.weight'] = weights_dict['vgg_local/conv2_2/weights:0'].permute(3,2,0,1)
weights['features.7.bias'] = weights_dict['vgg_local/conv2_2/biases:0']
weights['features.10.weight'] = weights_dict['vgg_local/conv3_1/weights:0'].permute(3,2,0,1)
weights['features.10.bias'] = weights_dict['vgg_local/conv3_1/biases:0']
weights['features.12.weight'] = weights_dict['vgg_local/conv3_2/weights:0'].permute(3,2,0,1)
weights['features.12.bias'] = weights_dict['vgg_local/conv3_2/biases:0']
weights['features.14.weight'] = weights_dict['vgg_local/conv3_3/weights:0'].permute(3,2,0,1)
weights['features.14.bias'] = weights_dict['vgg_local/conv3_3/biases:0']
weights['features.17.weight'] = weights_dict['vgg_local/conv4_1/weights:0'].permute(3,2,0,1)
weights['features.17.bias'] = weights_dict['vgg_local/conv4_1/biases:0']
weights['features.19.weight'] = weights_dict['vgg_local/conv4_2/weights:0'].permute(3,2,0,1)
weights['features.19.bias'] = weights_dict['vgg_local/conv4_2/biases:0']
weights['features.21.weight'] = weights_dict['vgg_local/conv4_3/weights:0'].permute(3,2,0,1)
weights['features.21.bias'] = weights_dict['vgg_local/conv4_3/biases:0']
weights['features.24.weight'] = weights_dict['vgg_local/conv5_1/weights:0'].permute(3,2,0,1)
weights['features.24.bias'] = weights_dict['vgg_local/conv5_1/biases:0']
weights['features.26.weight'] = weights_dict['vgg_local/conv5_2/weights:0'].permute(3,2,0,1)
weights['features.26.bias'] = weights_dict['vgg_local/conv5_2/biases:0']
weights['features.28.weight'] = weights_dict['vgg_local/conv5_3/weights:0'].permute(3,2,0,1)
weights['features.28.bias'] = weights_dict['vgg_local/conv5_3/biases:0']
weights['classifier.0.weight'] = weights_dict['vgg_local/fc6/weights:0'].permute(3,2,0,1).contiguous().view(4096,-1)
weights['classifier.0.bias'] = weights_dict['vgg_local/fc6/biases:0']
weights['classifier.3.weight'] = weights_dict['vgg_local/fc7/weights:0'].permute(3,2,0,1).squeeze(3).squeeze(2)
weights['classifier.3.bias'] = weights_dict['vgg_local/fc7/biases:0']
weights['classifier.6.weight'] = weights_dict['vgg_local/fc8/weights:0'].permute(3,2,0,1).squeeze(3).squeeze(2)
weights['classifier.6.bias'] = weights_dict['vgg_local/fc8/biases:0'] 

torch.save(weights, "text_objseg_pretrained_torch_converted")
