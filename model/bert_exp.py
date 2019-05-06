import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models
import numpy as np
from pytorch_pretrained_bert import BertModel
from model.resnet_exp import ImageModuleResnet
from model.baseline import DeconvLayer  
from model.model_utils import generate_spatial_batch, conv_relu, conv, init_weights

#https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L1070
class BertEmbedder(nn.Module):
    def __init__(self):
        super(BertEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Get max length from mask
        max_length, idx = (attention_mask != 0).max(0)        
        max_length = max_length.nonzero()[-1].item() + 1
        
        # S A V E G P U !
        if max_length < input_ids.shape[1]:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        # Return [CLS] token embedding
        return pooled_output  

class BertImgSeg(nn.Module):
    def __init__(self):
        super(BertImgSeg, self).__init__()

        mlp_hidden=500
        self.text_features = BertEmbedder()
        self.bert_emb_size = 768

        self.img_features = ImageModuleResnet()
        for param in self.img_features.parameters():
            param.requires_grad = False

        self.mlp1 = conv_relu(kernel_size=1, stride=1, in_channels=1000 + self.bert_emb_size + 8, out_channels=256)

        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        self.deconv = DeconvLayer(kernel_size=64, stride=32, output_dim=mlp_hidden, bias=False, in_channels=256)

        self.mlp2 = nn.Sequential(conv(kernel_size=1, stride=1, in_channels=mlp_hidden, out_channels=1))

    def forward(self, inputs):
        img_input, text_input = inputs

        img_out = self.img_features(img_input)

        # Gives [1 x 768] embedding from B E R T!
        bert_ids, bert_mask, bert_token_starts = text_input
        text_out = self.text_features(bert_ids, attention_mask=bert_mask)

        # N C H W format
        featmap_H, featmap_W = img_out.size(2), img_out.size(3)

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

        # Series of linear layers to reduce dimensions
        mlp_out = self.mlp1(concat_out)

        # Final deconvolution to get the upsampled mask
        deconv_out = self.deconv(mlp_out)

        generated_mask = self.mlp2(deconv_out)

        return F.softmax(generated_mask)