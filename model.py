import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models
import numpy as np

def init_weights(m):
    if type(m) == nn.Conv2d :
        print("init conv")
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        print("init bn")
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        print("Init linear")
        torch.nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.GRU or type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if('weight' in name):
                print("initializing LSTM/GRU weight ", name)
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                print("bias init", name)
                torch.nn.init.constant_(param, 0.0)


def conv_relu(kernel_size, stride, in_channels, out_channels, padding=0, bias=True):
    #TODO: weight init
    layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                      nn.ReLU(inplace=True))
    return layer

def conv(kernel_size, stride, in_channels, out_channels, padding=0, bias=True):
    #TODO: weight init
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return layer
    
def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return torch.Tensor(spatial_batch_val)


class LanguageModule(nn.Module):
    def __init__(self, vocab_size, emb_size, num_lstm_layers, hidden_size):
        super(LanguageModule, self).__init__()

        # TODO: need implementation for similar function
        # TODO: padding index
        self.embedding = nn.Embedding(vocab_size, emb_size) #, padding_idx=??)

        # TODO init bias as 1: https://github.com/ramithp/text_objseg/blob/tensorflow-1.x-compatibility/models/lstm_net.py#L16
        # Not bi: https://github.com/ramithp/text_objseg/blob/tensorflow-1.x-compatibility/util/rnn.py#L57
        self.lstm = nn.LSTM(input_size=emb_size,
                        hidden_size=hidden_size,
                        num_layers=num_lstm_layers,
                        bidirectional=False, batch_first=True)

    def forward(self, input_seq):
        # Incoming is a bsz x seq_len
        # Assumes already padded
        bsz = len(input_seq)
        max_seq_len = len(input_seq[0])
        in_lens = [len(seq) for seq in input_seq]

        # Pad to bs x max_seq_len
        padded_seqs = rnn_utils.pad_sequence(input_seq, batch_first=True)

        # Retrieve embeddings it into bsz x seq_len x embedding_size; including for <pad> tokens
        embedded_seqs = self.embedding(input_seq)
        
        # Goes in as bsz x seq_len x embedding_dim
        # Comes out of this as bsz x seq_len x embedding_dim
        packed_seqs = rnn_utils.pack_padded_sequence(embedded_seqs, lengths=in_lens, batch_first=True)

        hidden = None # internally sets to 0 vector embeddings
        output_packed, (hidden, _) = self.lstm(packed_seqs, hidden)
        
        # Output is now seq_len x bsz x hidden_dim
        return hidden[-1]

class ImageModule(nn.Module):


    def __init__(self):
        super(ImageModule, self).__init__()

        # Freeze VGG weights for all layers except FC layers
        self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor = self.feature_extractor.features
        
        # Padding is 3x3 because after VGG16 layers, we get a 16x16 feature map. k=7 needs 5 to get "SAME" padding
        vgg_fc7_full_conv = nn.Sequential(conv_relu(kernel_size=7, stride=1, in_channels=512, out_channels=4096, padding=(3, 3)),
                                  conv_relu(kernel_size=1, stride=1, in_channels=4096, out_channels=4096))

        # Padding not needed. Just a 1x1
        vgg_fc8_full_conv = conv(kernel_size=1, stride=1, in_channels=4096, out_channels=1000)

        self.feature_extractor.add_module("vgg_fc7_full_conv", vgg_fc7_full_conv)
        self.feature_extractor.add_module("vgg_fc8_full_conv", vgg_fc8_full_conv)

        # Optionally switch off fine-tune
        # for name, param in vgg16.named_parameters():
        #     # Last FC is set up for weight load
        #     if 'classifier' not in name:
        #         param.requires_grad = False

    def forward(self, inputs):
        x  = self.feature_extractor(inputs)
        print(x.shape)
        return x


class DeconvLayer(nn.Module):
    def __init__(self, kernel_size, stride, output_dim, bias=False):
        super(DeconvLayer, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels=1, out_channels=output_dim, kernel_size=kernel_size, 
                                        stride=stride, bias=bias, padding=16)
        
    def forward(self, inp):
#         batch_size, input_dim, input_height, input_width = inp.shape
        return self.dconv(inp)

class ImgSegRefExpModel(nn.Module):
    def __init__(self, mlp_hidden, vocab_size, emb_size, lstm_hidden_size):
        super(ImgSegRefExpModel, self).__init__()
        self.text_features = LanguageModule(vocab_size=vocab_size, emb_size=emb_size, num_lstm_layers=1, hidden_size=lstm_hidden_size)

        self.img_features = ImageModule()

        self.mlp1 = conv_relu(kernel_size=1, stride=1, in_channels=1000+lstm_hidden_size+8, out_channels=mlp_hidden)
        self.mlp2 = conv_relu(kernel_size=1, stride=1, in_channels=mlp_hidden, out_channels=1)

        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        self.deconv = DeconvLayer(kernel_size=64, stride=32, output_dim=1, bias=False)

    def forward(self, inputs):
        img_input, text_input = inputs

        img_out = self.img_features(img_input)
        text_out = self.text_features(text_input)

        # N C H W format
        featmap_H, featmap_W = img_out.size(2), img_out.size(3)
        print(featmap_H, featmap_W)
        # bsz x hidden_dim
        N, D_text = text_out.size(0), text_out.size(1)

        # Tile the textual features with the image feature-maps
        text_out = text_out.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, featmap_H, featmap_W)        

        # Generate spatial features to learn co-ordinates
        spatial_feats = generate_spatial_batch(N, featmap_H, featmap_W).permute(0, 3, 1, 2)
        
        # Concat 3 sources of inputs
        # Output is of shape N x (D_text + D_img + D_spatial) x H x W
        concat_out = torch.cat([F.normalize(text_out, p=2, dim=1),
                             F.normalize(img_out, p=2, dim=1),
                             spatial_feats], dim=1)

        mlp_out = self.mlp1(concat_out)
        mlp_out = self.mlp2(mlp_out)
        
        print(mlp_out.shape)

        generated_mask = self.deconv(mlp_out)

        return generated_mask                