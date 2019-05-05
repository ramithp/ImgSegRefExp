import torch

################################################################################
# Parameters
################################################################################
# Data parameters

# Evaluation Param
score_thresh = 1e-9

# Model Params
T = 20
N = 1
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

#Training Params
pos_loss_mult = 1.
neg_loss_mult = 1.

start_lr = 0.01
lr_decay_step = 10000
lr_decay_rate = 0.1
weight_decay = 0.0005
momentum = 0.9
max_iter = 30000
n_epochs = 5

fix_convnet = False
vgg_dropout = False
mlp_dropout = False
vgg_lr_mult = 1.

cls_loss_avg = 0
avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
decay = 0.99

# Device
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

root = './exp-referit/'
image_dir = root + 'referit-dataset/images/'
mask_dir = root + 'referit-dataset/mask/'
query_file = root + 'data/referit_query_train.json'
bbox_file = root + 'data/referit_bbox.json'
imcrop_file = root + 'data/referit_imcrop.json'
imsize_file = root + 'data/referit_imsize.json'
vocab_file = root + 'data/vocabulary_referit.txt'
query_file_val = root + 'data/referit_query_val.json'
custom_test_set = root + 'data/query_dict_custom_testset.json'

# trained model
pretrained_wts = True
# pretrained_model_file = "./text_objseg_pretrained_torch_converted_with_lstm.pt"
pretrained_model_file = "./model_dict_0.pt"
vocab_file = './vocabulary_referit.txt'

# Use Mask-R-CNN resnet-101 weights
resnet_wts_file = './resnet101-mask-r-cnn.pth'