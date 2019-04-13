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
    
def tf_lstm_to_pt(tf_weight, tf_bias, input_size, hidden_size, alpha=0.5):
    rows, cols = tf_weight.shape
    assert rows == input_size + hidden_size
    input_weights = tf_weight[:input_size, :] # inp_size x 4*hidden_size
    hidden_weights = tf_weight[input_size:, :]
    input_bias = tf_bias.view(4, hidden_size)
    input_bias = input_bias[[0,2,1,3],:]
    input_bias = input_bias.view(4*hidden_size)
    hidden_bias = alpha * input_bias
    input_bias = (1- alpha) * input_bias
    input_weights = input_weights.t()
    input_weights = input_weights.view(4, hidden_size, input_size)
    input_weights = input_weights[[0,2,1,3], :,:]
    input_weights = input_weights.view(4*hidden_size, input_size)
    hidden_weights = hidden_weights.t().view(4, hidden_size, hidden_size)[[0,2,1,3], :,:].view(4*hidden_size, hidden_size)
    return input_weights, hidden_weights, input_bias, hidden_bias
    # reordered weights from i,g,f,o to i,f,g,o
    
weights = OrderedDict()
weights['img_features.feature_extractor.0.weight'] = weights_dict['vgg_local/conv1_1/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.0.bias'] = weights_dict['vgg_local/conv1_1/biases:0']
weights['img_features.feature_extractor.2.weight'] = weights_dict['vgg_local/conv1_2/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.2.bias'] = weights_dict['vgg_local/conv1_2/biases:0']
weights['img_features.feature_extractor.5.weight'] = weights_dict['vgg_local/conv2_1/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.5.bias'] = weights_dict['vgg_local/conv2_1/biases:0']
weights['img_features.feature_extractor.7.weight'] = weights_dict['vgg_local/conv2_2/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.7.bias'] = weights_dict['vgg_local/conv2_2/biases:0']
weights['img_features.feature_extractor.10.weight'] = weights_dict['vgg_local/conv3_1/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.10.bias'] = weights_dict['vgg_local/conv3_1/biases:0']
weights['img_features.feature_extractor.12.weight'] = weights_dict['vgg_local/conv3_2/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.12.bias'] = weights_dict['vgg_local/conv3_2/biases:0']
weights['img_features.feature_extractor.14.weight'] = weights_dict['vgg_local/conv3_3/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.14.bias'] = weights_dict['vgg_local/conv3_3/biases:0']
weights['img_features.feature_extractor.17.weight'] = weights_dict['vgg_local/conv4_1/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.17.bias'] = weights_dict['vgg_local/conv4_1/biases:0']
weights['img_features.feature_extractor.19.weight'] = weights_dict['vgg_local/conv4_2/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.19.bias'] = weights_dict['vgg_local/conv4_2/biases:0']
weights['img_features.feature_extractor.21.weight'] = weights_dict['vgg_local/conv4_3/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.21.bias'] = weights_dict['vgg_local/conv4_3/biases:0']
weights['img_features.feature_extractor.24.weight'] = weights_dict['vgg_local/conv5_1/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.24.bias'] = weights_dict['vgg_local/conv5_1/biases:0']
weights['img_features.feature_extractor.26.weight'] = weights_dict['vgg_local/conv5_2/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.26.bias'] = weights_dict['vgg_local/conv5_2/biases:0']
weights['img_features.feature_extractor.28.weight'] = weights_dict['vgg_local/conv5_3/weights:0'].permute(3,2,0,1)
weights['img_features.feature_extractor.28.bias'] = weights_dict['vgg_local/conv5_3/biases:0']
weights['img_features.feature_extractor.vgg_fc7_full_conv.0.0.weight'] = weights_dict['vgg_local/fc6/weights:0'].permute(3,2,0,1).contiguous().view(4096,-1)
weights['img_features.feature_extractor.vgg_fc7_full_conv.0.0.bias'] = weights_dict['vgg_local/fc6/biases:0']
weights['img_features.feature_extractor.vgg_fc7_full_conv.1.0.weight'] = weights_dict['vgg_local/fc7/weights:0'].permute(3,2,0,1).squeeze(3).squeeze(2)
weights['img_features.feature_extractor.vgg_fc7_full_conv.1.0.bias'] = weights_dict['vgg_local/fc7/biases:0']
weights['img_features.feature_extractor.vgg_fc8_full_conv.weight'] = weights_dict['vgg_local/fc8/weights:0'].permute(3,2,0,1).squeeze(3).squeeze(2)
weights['img_features.feature_extractor.vgg_fc8_full_conv.bias'] = weights_dict['vgg_local/fc8/biases:0']
weights['text_features.embedding.weight'] = weights_dict['word_embedding/embedding:0']

tf_weight = weights_dict['lstm_lang/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0'] 
tf_bias = weights_dict['lstm_lang/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0']
input_weights, hidden_weights, input_bias, hidden_bias = tf_lstm_to_pt(tf_weight, tf_bias, 1000, 1000, alpha=0.5)
weights['text_features.lstm.weight_ih_l0'] = input_weights
weights['text_features.lstm.weight_hh_l0'] = hidden_weights
weights['text_features.lstm.bias_ih_l0'] = input_bias 
weights['text_features.lstm.bias_hh_l0'] = hidden_bias

weights['mlp.0.weight'] = weights_dict['classifier/mlp_l1/weights:0'].permute(3,2,0,1)
weights['mlp.0.bias'] = weights_dict['classifier/mlp_l1/biases:0']
weights['mlp.1.weight'] = weights_dict['classifier/mlp_l2/weights:0'].permute(3,2,0,1)
weights['mlp.1.bias'] = weights_dict['classifier/mlp_l2/biases:0']

weights['deconv.dconv.weight'] = weights_dict['classifier/upsample32s/weights:0'].permute(3,2,0,1)

torch.save(weights, "text_objseg_pretrained_torch_converted")
