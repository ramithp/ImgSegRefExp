import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models
import numpy as np
from model.baseline import LanguageModule, DeconvLayer
from model.resnet_exp import ImageModuleResnet
from model.model_utils import generate_spatial_batch, conv_relu, conv, init_weights
import config
from baseline import VisualContextModule

class ResAttnImgSeg(nn.Module):
    def __init__(self, mlp_hidden, vocab_size, emb_size, lstm_hidden_size):
        super(ResAttnImgSeg, self).__init__()
        self.text_features = LanguageModule(vocab_size=vocab_size, emb_size=emb_size, num_lstm_layers=1,
                                            hidden_size=lstm_hidden_size, return_all=True)

        self.img_features = ImageModuleResnet(resnet_weights_file=config.resnet_wts_file)

        img_feats_channels = 1000
        Cg = 256
        
        self.mlp1 = conv_relu(kernel_size=1, stride=1, in_channels=img_feats_channels + lstm_hidden_size + 8 + Cg, out_channels=mlp_hidden)
        self.mlp2 = nn.Sequential(conv(kernel_size=1, stride=1, in_channels=mlp_hidden, out_channels=1))
        
        self.visual_context = VisualContextModule(Cg,img_feats_channels + 8)

        # init
        self.mlp1.apply(init_weights)
        self.mlp2.apply(init_weights)

        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        self.deconv = DeconvLayer(kernel_size=64, stride=32, output_dim=1, bias=False)

        self.attn_vec_size = 128
        self.attn_key_lin = nn.Linear(img_feats_channels, 128)
        self.attn_queries_lin = nn.Linear(lstm_hidden_size, 128)

    def get_mask(self, input_lens):
        #         pdb.set_trace()
        batch_size = input_lens.shape[0]
        max_inp_len = 20  # torch.max(input_lens).item()
        #         pdb.set_trace()
        out_mod = torch.arange(max_inp_len).cuda().unsqueeze(0).repeat([batch_size, 1])
        out = out_mod < (max_inp_len - input_lens.unsqueeze(1))
        return out.byte()

    def get_attn_weights(self, keys, queries):
        return torch.matmul(keys, queries.unsqueeze(1).permute(0, 1, 3, 2))

    def get_attn_vec(self, lstm_output, text_lens, img_feats):
        # lstm_output: B x T x hdim, img_feats: B x 16 x 16 x 1000
        # compute alpha (B x H x W x T)
        # gen keys, values
        keys = self.attn_key_lin(img_feats)  # B x H x W x attn
        queries = self.attn_queries_lin(lstm_output)  # B x T x attn
        alphas = self.get_attn_weights(keys, queries)  # B x H x W x T

        mask = self.get_mask(text_lens)
        # pdb.set_trace()
        alphas.masked_fill_(mask.unsqueeze(1).unsqueeze(1), -float("inf"))  # fills -inf where mask is 1
        weights_over_text = F.softmax(alphas, dim=-1)  # B x H x W x T
        att_vecs = torch.matmul(weights_over_text, lstm_output.unsqueeze(1))  # B x H x W x hdim
        return att_vecs, weights_over_text

    def forward(self, inputs):
        img_input, text_input = inputs

        img_out = self.img_features(img_input)
        text_out, text_lens = self.text_features(text_input)

        # N C H W format
        featmap_H, featmap_W = img_out.size(2), img_out.size(3)

        # bsz x hidden_dim
        # N, D_text = text_out.size(0), text_out.size(1)
        N, T, D_text = text_out.shape

        # Tile the textual features with the image feature-maps
        # text_out = text_out.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, featmap_H, featmap_W)

        text_out, weights_over_text = self.get_attn_vec(text_out, text_lens, img_out.permute(0, 2, 3, 1))
        #         pdb.set_trace()
        # B x H x W x hdim
        text_out = text_out.permute( 0, 3, 1, 2)

        spatial_feats = generate_spatial_batch(N, featmap_H, featmap_W).permute(0, 3, 1, 2)
        visual_context = self.visual_context.forward(torch.cat([F.normalize(img_out, p=2, dim=1),
                                spatial_feats], dim=1),weights_over_text.permute(0,3,1,2))

        # Generate spatial features to learn co-ordinates
        
        #         pdb.set_trace()
        # Concat 3 sources of inputs
        # Output is of shape N x (D_text + D_img + D_spatial + Cg) x H x W
        concat_out = torch.cat([F.normalize(text_out, p=2, dim=1),
                                F.normalize(img_out, p=2, dim=1),
                                spatial_feats,visual_context], dim=1)

        # Series of linear layers to reduce dimensions
        mlp_out = self.mlp1(concat_out)
        mlp_out = self.mlp2(mlp_out)

        # Final deconvolution to get the upsampled mask
        generated_mask = self.deconv(mlp_out)

        return generated_mask
