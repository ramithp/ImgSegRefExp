    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

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

class LSTMLayer(nn.Module):
    def __init__(self, vocab_size, emb_size, pretrained_path, num_lstm_layers):
        super(PhonemePredictor, self).__init__()


    # embedding matrix with each row containing the embedding vector of a word
    # this has to be done on CPU currently
    with tf.variable_scope('word_embedding'), tf.device("/cpu:0"):
        embedding_mat = tf.get_variable("embedding", [num_vocab, embed_dim])
        # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)

        lstm_top = lstm('lstm_lang', embedded_seq, None, output_dim=lstm_dim,
                    num_layers=1, forget_bias=1.0, apply_dropout=False,
                    concat_output=False)[-1]

    # TODO: need implementation for similar function
    self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=??)
    data.load_pretrained_embedding(self.embedding, pretrained_path)

    # TODO init bias as 1: https://github.com/ramithp/text_objseg/blob/tensorflow-1.x-compatibility/models/lstm_net.py#L16
    # Not bi: https://github.com/ramithp/text_objseg/blob/tensorflow-1.x-compatibility/util/rnn.py#L57
    self.lstm = nn.LSTM(input_size=emb_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_lstm_layers,
                    bidirectional=False)

    def forward(self, input_seq):
        # Incoming is a bs x seq_len
        # Assumes already padded
        bsz = len(input_seq)
        max_seq_len =len(input_seq[0])
        in_lens = [len(seq) for seq in input_seq]

        # Pad to bs x max_seq_len
        padded_seqs = rnn_utils.pad_sequence(input_seq)
        padded_seqs = padded_seqs

        # Retrieve embeddings it into bsz x seq_len x embedding_size; including for <pad> tokens
        embedded_seqs = self.embedding(input_seq)
        
        # Goes in as bsz x seq_len x embedding_dim
        # Comes out of this as seq_len x bsz x embedding_dim
        packed_seqs = rnn_utils.pack_padded_sequence(embedded_seqs, lengths=in_lens)

        # TODO: do something better than this
        hidden = None # internally sets to 0 vector embeddings
        output_packed, hidden = self.lstm (packed_seqs, hidden)

        # Unpack sequence to padded level to use in linear layer
        # Comes out as seq_len x bsz x hidden_dim, because we used batch_first=True
        output_padded, _ = rnn_utils.pad_packed_sequence(output_packed)

        # Output is now seq_len x bsz x hidden_dim
        return output_padded

class CNNLayer(nn.Module):

    @staticmethod
    def conv_1x1_bn_relu(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True))

    @staticmethod
    def conv_1x1_relu(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True))


    def conv_relu_layer(bottom, kernel_size, stride, output_dim, padding='SAME',
                    bias_term=True, weights_initializer=None, biases_initializer=None):
    conv = conv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                      bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(conv)
    return relu


    def __init__(self, pretrained_wt_path):
        super(CNNLayer, self).__init__()

        # Freeze VGG weights for all layers except FC layers
        vgg16 = models.vgg16(pretrained=True)

        for name, param in vgg16.named_parameters():
            # Last FC is set up for weight load
            if 'classifier' not in name:
                param.requires_grad = False

        # TODO weight load function
        data_utils.load_vgg16_fc(vgg16, pretrained_wt_path)

    def forward(self, seq_batch):
        return outputs

class ImgSegRefExpModel(nn.Module):
    def __init__(self):
        super(ImgSegRefExpModel, self).__init__()
        self.lstm = LSTMLayer(
            vocab_size=,
            emb_size=, 
            pretrained_path=, 
            num_lstm_layers=1)

        self.cnn = CNNLayer(pretrained_wt_path=)

        # TODO
        self.mlp = ???

        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        self.deconv = ??? 

    def forward(self, seq_batch):
        return seq_batch