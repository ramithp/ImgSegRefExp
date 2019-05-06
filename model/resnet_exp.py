import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models
import numpy as np
from model.baseline import LanguageModule, DeconvLayer
from model.model_utils import generate_spatial_batch, conv_relu, conv, init_weights
import config

class ImageModuleResnet(nn.Module):
    def __init__(self, resnet_weights_file=None):
        super(ImageModuleResnet, self).__init__()

        # Freeze Resnet weights for all layers except FC layers
        self.feature_extractor = models.resnet101(pretrained=False)
        # if resnet_weights_file:
        #     self.feature_extractor.load_state_dict(torch.load(resnet_weights_file))
        #     print("Loaded weights from resnet weights file")

        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-2])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Padding is 3x3 because after VGG16 layers, we get a 16x16 feature map. k=7 needs 5 to get "SAME" padding
        self.res_fc7_full_conv = nn.Sequential(
            conv_relu(kernel_size=1, stride=1, in_channels=2048, out_channels=1024),
            conv_relu(kernel_size=7, stride=1, in_channels=1024, out_channels=2048, padding=(3, 3)),
            conv_relu(kernel_size=1, stride=1, in_channels=2048, out_channels=2048))


        # Padding not needed. Just a 1x1
        self.res_fc8_full_conv = conv(kernel_size=1, stride=1, in_channels=2048, out_channels=1000)

        # init
        self.res_fc7_full_conv.apply(init_weights)
        self.res_fc8_full_conv.apply(init_weights)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.res_fc7_full_conv(x)
        x = self.res_fc8_full_conv(x)
        return x

class ResImgSeg(nn.Module):
    def __init__(self, mlp_hidden, vocab_size, emb_size, lstm_hidden_size):
        super(ResImgSeg, self).__init__()
        self.text_features = LanguageModule(vocab_size=vocab_size, emb_size=emb_size, num_lstm_layers=1,
                                            hidden_size=lstm_hidden_size)

        self.img_features = ImageModuleResnet(resnet_weights_file=config.resnet_wts_file)

        self.mlp1 = conv_relu(kernel_size=1, stride=1, in_channels=1000 + lstm_hidden_size + 8, out_channels=mlp_hidden)
        self.mlp2 = nn.Sequential(conv(kernel_size=1, stride=1, in_channels=mlp_hidden, out_channels=1))
        
        # init
        self.mlp1.apply(init_weights)
        self.mlp2.apply(init_weights)

        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        self.deconv = DeconvLayer(kernel_size=64, stride=32, output_dim=1, bias=False)

    def forward(self, inputs):
        img_input, text_input = inputs

        img_out = self.img_features(img_input)
        text_out = self.text_features(text_input)

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
        mlp_out = self.mlp2(mlp_out)

        # Final deconvolution to get the upsampled mask
        generated_mask = self.deconv(mlp_out)

        return generated_mask

class ResImgSegDeconved(nn.Module):
    def __init__(self, mlp_hidden, vocab_size, emb_size, lstm_hidden_size):
        super(ResImgSegDeconved, self).__init__()
        self.text_features = LanguageModule(vocab_size=vocab_size, emb_size=emb_size, num_lstm_layers=1,
                                            hidden_size=lstm_hidden_size)
        self.img_features = ImageModuleResnet(resnet_weights_file=config.resnet_wts_file)
        self.mlp1 = conv_relu(kernel_size=1, stride=1, in_channels=1000 + lstm_hidden_size + 8, out_channels=mlp_hidden)
        self.mlp22 = nn.Sequential(conv(kernel_size=1, stride=1, in_channels=mlp_hidden, out_channels=32))
        self.mlp3 = conv(kernel_size=1, stride=1, in_channels=32, out_channels=1)
        # init
        self.mlp1.apply(init_weights)
        self.mlp22.apply(init_weights)
        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        self.deconv_large = DeconvLayer(kernel_size=64, stride=32, output_dim=32,in_channels=32, bias=False)
    def forward(self, inputs):
        img_input, text_input = inputs
        img_out = self.img_features(img_input)
        text_out = self.text_features(text_input)
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
        mlp_out = self.mlp22(mlp_out)
        # Final deconvolution to get the upsampled mask
        generated_mask = self.deconv_large(mlp_out)
        return self.mlp3(generated_mask)
