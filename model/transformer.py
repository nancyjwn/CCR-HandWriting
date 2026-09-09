import time
import math
import copy
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

from config import config
from util import get_alphabet

alphabet = get_alphabet(config['mode'])


class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        # x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        # x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len, :].to(x.device)
        # return positional encoding (same shape as x) - note: original code concatenated later,
        # here we just return the pe slice so caller can decide concat or add.
        return pe_slice


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        if compress_attention:
            self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        if mask is not None:
            # mask expected shape (batch, seq_len, seq_len) -> make (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1).to(query.device)
        nbatches = query.size(0)

        # project and split heads
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, attention_map = attention(query, key, value, mask=mask, dropout=self.dropout, align=align)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    # returns (1, size, size) boolean mask where True = allowed, False = masked
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask shape expected broadcastable to scores
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # decoder works on 1024-dim tokens (512 word embed + 512 pos embed)
        self.mask_multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        self.multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1, compress_attention=True)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

    def forward(self, text, conv_feature):
        # text: (batch, seq_len, 1024)
        text_max_length = text.shape[1]
        # build mask on device of text
        mask = subsequent_mask(text_max_length).to(text.device)

        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None)
        result = self.mul_layernorm2(result + word_image_align)

        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class Transformer(nn.Module):

    def __init__(self, mode):
        super(Transformer, self).__init__()

        self.mode = mode
        self.word_n_class = len(alphabet)
        # word embedding dim 512
        self.embedding_word = Embeddings(512, self.word_n_class)
        # positional encoding produces 512-d pe
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)

        # encoder will be moved by caller (do not call .cuda() here)
        self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3])
        self.decoder = Decoder()
        # generator expects 1024-d hidden (512+512)
        self.generator_word = Generator(1024, self.word_n_class)
        self.attribute = None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, text_length, text_input, conv_feature=None, test=False):
        """
        image: (batch, 3, H, W)
        text_length: None or tensor [batch] (lengths)
        text_input: tensor (batch, seq_len) of token ids
        conv_feature: optional conv features (b, c, h, w)
        test: if True, returns predictions per position (batch, seq_len, vocab)
        """

        # compute conv features if not provided
        if conv_feature is None:
            conv_feature = self.encoder(image)

        # if no text requested, return conv features (used in some flows)
        if text_length is None:
            return {
                'conv': conv_feature,
            }

        # embedding (batch, seq_len, 512)
        text_embedding = self.embedding_word(text_input)  # (B, L, 512)
        # positional encoding (512) of size same as text_embedding, placed on same device
        pos_embed = self.pe(torch.zeros_like(text_embedding).to(text_embedding.device))  # (1, L, 512) broadcastable
        # concatenate to make 1024-d tokens (512 word emb + 512 pos emb)
        # pos_embed currently shape (1, L, 512) -> expand to batch
        pos_embed = pos_embed.expand(text_embedding.size(0), -1, -1)
        text_input_with_pe = torch.cat([text_embedding, pos_embed], dim=2)  # (B, L, 1024)

        # decode with attention (decoder expects 1024-d tokens)
        text_input_with_pe, attention_map = self.decoder(text_input_with_pe, conv_feature)
        word_decoder_result = self.generator_word(text_input_with_pe)  # (B, L, vocab)

        if test:
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                'conv': conv_feature,
            }
        else:
            # total number of tokens across batch
            total_length = int(torch.sum(text_length).item())
            device = word_decoder_result.device
            dtype = word_decoder_result.dtype
            probs_res = torch.zeros(total_length, self.word_n_class, device=device, dtype=dtype)

            start = 0
            for index, length in enumerate(text_length):
                length_i = int(length.item())
                probs_res[start:start + length_i, :] = word_decoder_result[index, 0:0 + length_i, :]
                start = start + length_i

            return {
                'pred': probs_res,
                'map': attention_map,
                'conv': conv_feature,
            }


if __name__ == '__main__':
    # quick smoke test (CPU) - caller can move model to GPU by model.to(device)
    net = ResNet(num_in=3, block=BasicBlock, layers=[1, 2, 5, 3])
    image = torch.randn(8, 3, 64, 64)
    result = net(image)
    print(result.shape)
    pass