#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pysam
import csv
import numpy as np
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader
from gensim.models import Word2Vec


# In[1]:


#lb, ub of len of DEL, DUP
min_len = 50
max_len = 2000
max_len_ins = 500

#upper bound of cov (from select_seq.ipynb)
cov_ub = 300

#the interval of two sv to be counted as overlap
ol_interval = 1000

#flanking length
flanking = 200
#flanking length
flanking_ins = 500

# ref fasta file
fasta_file_path = "/scratch1/jianzhiy/svpred/reference/hg38.no_alts.fasta"

# no of features: cov, sc, insert size, 3mer emb
no_features = 4

# add constant to scale features
cov_scaler = 50
sc_scaler = 10
insert_scaler = 10000
emb_scaler = 1

# upper bound of insert size after scaling
insert_upper = 50

# the pretrained word2vec model
kmer_emb_size = 5
# Word2Vec_model_name = "del_2000_30samples_3mers_truth__word2vec"
# Word2Vec_model = Word2Vec.load(f"{Word2Vec_model_name}.model")


# In[6]:


kmer_mapping = dict()
kmer_mapping['AAA'] = 1
kmer_mapping['AAG'] = 2
kmer_mapping['AAC'] = 3
kmer_mapping['AAT'] = 4

kmer_mapping['AGA'] = 5
kmer_mapping['AGG'] = 6
kmer_mapping['AGC'] = 7
kmer_mapping['AGT'] = 8

kmer_mapping['ACA'] = 9
kmer_mapping['ACG'] = 10
kmer_mapping['ACC'] = 11
kmer_mapping['ACT'] = 12

kmer_mapping['ATA'] = 13
kmer_mapping['ATG'] = 14
kmer_mapping['ATC'] = 15
kmer_mapping['ATT'] = 16

kmer_mapping['GAA'] = 17
kmer_mapping['GAG'] = 18
kmer_mapping['GAC'] = 19
kmer_mapping['GAT'] = 20

kmer_mapping['GGA'] = 21
kmer_mapping['GGG'] = 22
kmer_mapping['GGC'] = 23
kmer_mapping['GGT'] = 24

kmer_mapping['GCA'] = 25
kmer_mapping['GCG'] = 26
kmer_mapping['GCC'] = 27
kmer_mapping['GCT'] = 28

kmer_mapping['GTA'] = 29
kmer_mapping['GTG'] = 30
kmer_mapping['GTC'] = 31
kmer_mapping['GTT'] = 32

kmer_mapping['CAA'] = 33
kmer_mapping['CAG'] = 34
kmer_mapping['CAC'] = 35
kmer_mapping['CAT'] = 36

kmer_mapping['CGA'] = 37
kmer_mapping['CGG'] = 38
kmer_mapping['CGC'] = 39
kmer_mapping['CGT'] = 40

kmer_mapping['CCA'] = 41
kmer_mapping['CCG'] = 42
kmer_mapping['CCC'] = 43
kmer_mapping['CCT'] = 44

kmer_mapping['CTA'] = 45
kmer_mapping['CTG'] = 46
kmer_mapping['CTC'] = 47
kmer_mapping['CTT'] = 48

kmer_mapping['TAA'] = 49
kmer_mapping['TAG'] = 50
kmer_mapping['TAC'] = 51
kmer_mapping['TAT'] = 52

kmer_mapping['TGA'] = 53
kmer_mapping['TGG'] = 54
kmer_mapping['TGC'] = 55
kmer_mapping['TGT'] = 56

kmer_mapping['TCA'] = 57
kmer_mapping['TCG'] = 58
kmer_mapping['TCC'] = 59
kmer_mapping['TCT'] = 60

kmer_mapping['TTA'] = 61
kmer_mapping['TTG'] = 62
kmer_mapping['TTC'] = 63
kmer_mapping['TTT'] = 20


# In[7]:


kmer_reverse_mapping = dict()
for key, value in kmer_mapping.items():
    kmer_reverse_mapping[value] = key


# In[3]:


# chr
chr_list = ["chr1", "chr2", "chr3", "chr4", "chr5",
            "chr6", "chr7", "chr8", "chr9", "chr10",
            "chr11", "chr12", "chr13", "chr14", "chr15",
            "chr16", "chr17", "chr18", "chr19", "chr20",
            "chr21", "chr22"]

# valid types
# valid_types = ['DEL', 'INS', 'DUP']
valid_types = ['DEL', 'INS', 'DUP', 'INV']

#len of chr
chr_len_ori = [250000000, 244000000, 202000000, 194000000, 183000000, 
            173000000, 161000000, 147000000, 151000000, 136000000, 
            136000000, 134000000, 116000000, 108000000, 103000000, 
            96400000, 84300000, 80600000, 61800000, 66300000, 
            48200000, 51400000, 157000000, 62500000]
chr_len = [int(0.8 * length) for length in chr_len_ori]


# In[5]:


# # Define the Transformer model
# class TransformerClassifier_naive(nn.Module):
#     def __init__(self, d_model, nhead, num_layers, num_classes, dim_feedforward):
#         super(TransformerClassifier_naive, self).__init__()
        
#         self.embedding = nn.Embedding(512, d_model)
#         self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
#         self.fc = nn.Linear(d_model, num_classes)
        
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#         x = self.transformer.encoder(x)
#         x = x.permute(1, 0, 2)
#         x = self.fc(x[:, 0, :])
#         return x


# In[ ]:


# # Define the Transformer model
# class TransformerClassifier(nn.Module):    
#     def __init__(self, d_model, nhead, num_layers, num_classes, dim_feedforward, dropout=0.1):
#         super(TransformerClassifier, self).__init__()
#         self.embedding = nn.Embedding(512, d_model)
#         self.linear0 = nn.Linear(2, d_model)
#         # self.pos_encoder = self.make_sincos_pos_encoding(d_model, 2000)
#         self.pos_encoder = PositionalEncoding(d_model, (max_len + 2 * flanking + 100) * 2)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(d_model, num_classes)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc_out.weight.data.uniform_(-initrange, initrange)

#     def make_sincos_pos_encoding(self, d_model, max_len):
#         pos_encoding = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pos_encoding[:, 0::2] = torch.sin(position * div_term)
#         pos_encoding[:, 1::2] = torch.cos(position * div_term)
#         pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
#         return pos_encoding

#     def forward(self, src):
#         # print(src.size())
#         # src = self.linear0(src) 
        
#         src = self.embedding(src.long())  # Embedding each integer
#         src = src.view(src.size(0), -1, src.size(-1))
        
#         src = self.pos_encoder(src)
        
#         # print(src.size())
#         src = src.permute(1, 0, 2)  # Adjust shape for transformer encoder: [seq_len, batch_size, d_model]
#         # src += self.pos_encoder[:src.size(0), :, :].to(src.device)
#         transformer_output = self.transformer_encoder(src)
#         # print("1", transformer_output.size())
#         output = transformer_output.mean(dim=0)
#         output = self.fc_out(output)
#         return output


# In[ ]:


# sparse attention

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    # print(x.size())
    # x = torch.transpose(x, 0, 2, 1, 3)
    x = torch.permute(x, (0, 2, 1, 3))
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return torch.transpose(split_states(x, n), 0, 2, 1, 3)

def merge_heads(x):
    return merge_states(torch.transpose(x, 0, 2, 1, 3))

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)


    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    # bst = get_blocksparse_obj(n_ctx, heads, attn_mode, blocksize, local_attn_ctx, num_verts, vertsize)
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    # w = bst.masked_softmax(w, scale=scale_amount)
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        # a = torch.transpose(a, 0, 2, 1, 3)
        a = torch.permute(a, (0, 2, 1, 3))
        a = torch.reshape(a, [n, t, embd])
    return a

class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx, self.blocksize)


# In[ ]:


class BinaryClassificationTransformer11117Norm2(nn.Module):
    def __init__(self, d_model, nhead, attn_mode, local_attn_ctx=None, blocksize=32, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(BinaryClassificationTransformer11117Norm2, self).__init__()
        self.embedding = nn.Embedding(512, d_model)
        self.linear0 = nn.Linear(8, d_model)
        self.pos_encoder = PositionalEncoding(d_model, (max_len + 2 * flanking + 100) * 2)
        self.encoder_layer = TransformerEncoderLayerWithSparseAttentionNorm1(d_model, nhead, attn_mode, local_attn_ctx, blocksize, dim_feedforward, dropout)
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, src):
        # print(src.size())
        src = self.linear0(src) 
        # src = self.embedding(src.long())  # Embedding each integer
        # src = src.view(src.size(0), src.size(1),  -1)
        
        src = self.pos_encoder(src)
        # print(src.size())
        # print(src.size())
        src = src.permute(1, 0, 2)
        encoded = self.encoder(src)
        # print("1", encoded.size())
        # encoded = encoded.permute(1, 0, 2)
        output = encoded.mean(dim=0)
        # print(output.size())
        output = self.classifier(output)
        # print(output.size())
        return output


# In[ ]:


class TransformerEncoderLayerWithSparseAttention(nn.Module):
    def __init__(self, d_model, heads, attn_mode, local_attn_ctx=None, blocksize=32, dim_feedforward=32, dropout=0.1):
        super(TransformerEncoderLayerWithSparseAttention, self).__init__()
        self.sparse_attention = SparseAttention(heads=heads, attn_mode=attn_mode, local_attn_ctx=local_attn_ctx, blocksize=blocksize)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src):
        # print("1", src.size())
        src = src.permute(1, 0, 2)
        src2 = self.sparse_attention(q=src, k=src, v=src)
        # print("2", src.size())
        src = src + self.dropout1(src2)
        # print("3", src.size())
        # src = self.norm1(src)
        # print("4", src.size())
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # print("5", src.size())
        # src = src + self.dropout2(src2)
        # print("6", src.size())
        # src = self.norm2(src)
        # print("7", src.size())
        src = src.permute(1, 0, 2)
        return src

class TransformerEncoderLayerWithSparseAttentionNorm(nn.Module):
    def __init__(self, d_model, heads, attn_mode, local_attn_ctx=None, blocksize=32, dim_feedforward=32, dropout=0.1):
        super(TransformerEncoderLayerWithSparseAttentionNorm, self).__init__()
        self.sparse_attention = SparseAttention(heads=heads, attn_mode=attn_mode, local_attn_ctx=local_attn_ctx, blocksize=blocksize)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src):
        # print("1", src.size())
        src = src.permute(1, 0, 2)
        src2 = self.sparse_attention(q=src, k=src, v=src)
        # print("2", src.size())
        src = src + self.dropout1(src2)
        # print("3", src.size())
        src = self.norm1(src)
        # print("4", src.size())
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # print("5", src.size())
        # src = src + self.dropout2(src2)
        # print("6", src.size())
        # src = self.norm2(src)
        # print("7", src.size())
        src = src.permute(1, 0, 2)
        return src

class BinaryClassificationTransformer11117Norm(nn.Module):
    def __init__(self, d_model, nhead, attn_mode, local_attn_ctx=None, blocksize=32, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(BinaryClassificationTransformer11117Norm, self).__init__()
        self.embedding = nn.Embedding(512, d_model)
        self.linear0 = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model * 2, (max_len + 2 * flanking + 100) * 2)
        self.encoder_layer = TransformerEncoderLayerWithSparseAttentionNorm(d_model * 2, nhead, attn_mode, local_attn_ctx, blocksize, dim_feedforward, dropout)
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, src):
        # print(src.size())
        # src = self.linear0(src) 
        
        src = self.embedding(src.long())  # Embedding each integer
        src = src.view(src.size(0), src.size(1),  -1)
        
        src = self.pos_encoder(src)
        # print(src.size())
        # print(src.size())
        src = src.permute(1, 0, 2)
        encoded = self.encoder(src)
        # print("1", encoded.size())
        # encoded = encoded.permute(1, 0, 2)
        output = encoded.mean(dim=0)
        # print(output.size())
        output = self.classifier(output)
        # print(output.size())
        return output

class TransformerEncoderLayerWithSparseAttentionNorm1(nn.Module):
    def __init__(self, d_model, heads, attn_mode, local_attn_ctx=None, blocksize=32, dim_feedforward=32, dropout=0.1):
        super(TransformerEncoderLayerWithSparseAttentionNorm1, self).__init__()
        self.sparse_attention = SparseAttention(heads=heads, attn_mode=attn_mode, local_attn_ctx=local_attn_ctx, blocksize=blocksize)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src):
        # print("1", src.size())
        src = src.permute(1, 0, 2)
        src2 = self.sparse_attention(q=src, k=src, v=src)
        # print("2", src.size())
        src = src + self.dropout1(src2)
        # print("3", src.size())
        src = self.norm1(src)
        # print("4", src.size())
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # print("5", src.size())
        src = src + self.dropout2(src2)
        # print("6", src.size())
        src = self.norm2(src)
        # print("7", src.size())
        src = src.permute(1, 0, 2)
        return src

class BinaryClassificationTransformer11117Norm1(nn.Module):
    def __init__(self, d_model, nhead, attn_mode, local_attn_ctx=None, blocksize=32, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(BinaryClassificationTransformer11117Norm1, self).__init__()
        self.embedding = nn.Embedding(512, d_model)
        self.linear0 = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model * 2, (max_len + 2 * flanking + 100) * 2)
        self.encoder_layer = TransformerEncoderLayerWithSparseAttentionNorm1(d_model * 2, nhead, attn_mode, local_attn_ctx, blocksize, dim_feedforward, dropout)
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, src):
        # print(src.size())
        # src = self.linear0(src) 
        
        src = self.embedding(src.long())  # Embedding each integer
        src = src.view(src.size(0), src.size(1),  -1)
        
        src = self.pos_encoder(src)
        # print(src.size())
        # print(src.size())
        src = src.permute(1, 0, 2)
        encoded = self.encoder(src)
        # print("1", encoded.size())
        # encoded = encoded.permute(1, 0, 2)
        output = encoded.mean(dim=0)
        # print(output.size())
        output = self.classifier(output)
        # print(output.size())
        return output

class BinaryClassificationTransformerNorm1(nn.Module):
    def __init__(self, d_model, nhead, attn_mode, local_attn_ctx=None, blocksize=32, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(BinaryClassificationTransformerNorm1, self).__init__()
        self.embedding = nn.Embedding(512, d_model)
        self.linear0 = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, (max_len + 2 * flanking + 100) * 2)
        self.encoder_layer = TransformerEncoderLayerWithSparseAttentionNorm1(d_model, nhead, attn_mode, local_attn_ctx, blocksize, dim_feedforward, dropout)
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, src):
        # print(src.size())
        # src = self.linear0(src) 
        
        src = self.embedding(src.long())  # Embedding each integer
        src = src.view(src.size(0), -1, src.size(-1))
        
        src = self.pos_encoder(src)
        # print(src.size())
        # print(src.size())
        src = src.permute(1, 0, 2)
        encoded = self.encoder(src)
        # print("1", encoded.size())
        # encoded = encoded.permute(1, 0, 2)
        output = encoded.mean(dim=0)
        # print(output.size())
        output = self.classifier(output)
        # print(output.size())
        return output

class BinaryClassificationTransformer(nn.Module):
    def __init__(self, d_model, nhead, attn_mode, local_attn_ctx=None, blocksize=32, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(BinaryClassificationTransformer, self).__init__()
        self.embedding = nn.Embedding(512, d_model)
        self.linear0 = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, (max_len + 2 * flanking + 100) * 2)
        self.encoder_layer = TransformerEncoderLayerWithSparseAttention(d_model, nhead, attn_mode, local_attn_ctx, blocksize, dim_feedforward, dropout)
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, src):
        # print(src.size())
        # src = self.linear0(src) 
        
        src = self.embedding(src.long())  # Embedding each integer
        src = src.view(src.size(0), -1, src.size(-1))
        
        src = self.pos_encoder(src)
        # print(src.size())
        # print(src.size())
        src = src.permute(1, 0, 2)
        encoded = self.encoder(src)
        # print("1", encoded.size())
        # encoded = encoded.permute(1, 0, 2)
        output = encoded.mean(dim=0)
        # print(output.size())
        output = self.classifier(output)
        # print(output.size())
        return output

class BinaryClassificationTransformer11117(nn.Module):
    def __init__(self, d_model, nhead, attn_mode, local_attn_ctx=None, blocksize=32, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(BinaryClassificationTransformer11117, self).__init__()
        self.embedding = nn.Embedding(512, d_model)
        self.linear0 = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model * 2, (max_len + 2 * flanking + 100) * 2)
        self.encoder_layer = TransformerEncoderLayerWithSparseAttention(d_model * 2, nhead, attn_mode, local_attn_ctx, blocksize, dim_feedforward, dropout)
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, src):
        # print(src.size())
        # src = self.linear0(src) 
        
        src = self.embedding(src.long())  # Embedding each integer
        src = src.view(src.size(0), src.size(1),  -1)
        
        src = self.pos_encoder(src)
        # print(src.size())
        # print(src.size())
        src = src.permute(1, 0, 2)
        encoded = self.encoder(src)
        # print("1", encoded.size())
        # encoded = encoded.permute(1, 0, 2)
        output = encoded.mean(dim=0)
        # print(output.size())
        output = self.classifier(output)
        # print(output.size())
        return output


# In[ ]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough PE matrix of zeros
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension (B x SeqLen x DModel)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, d_model]
        """
        # Add positional encoding to the input batch, assuming x is of shape (B, SeqLen, DModel)
        x = (x + self.pe[:, :x.size(1)]).cuda()
        return x


# In[ ]:


class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape
        output = torch.zeros_like(x)

        for i in range(seq_len):
            # Determine the window range
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            
            # Apply linear transformation within the window
            window_features = self.linear(x[:, start:end, :])
            # Sum features within the window for simplicity
            output[:, i, :] = window_features.sum(dim=1)
        
        return output


# In[2]:


class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.window_size = window_size

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        attention_scores = torch.zeros((N, self.heads, query_len, self.window_size * 2 + 1), device=query.device)

        for i in range(query_len):
            # Define the window bounds
            start = max(0, i - self.window_size)
            end = min(key_len, i + self.window_size + 1)

            # Calculate scores
            scores = torch.einsum("nhqd,nhkd->nhqk", [queries[:, i:i+1, :, :], keys[:, start:end, :, :]])
            attention_scores[:, :, i, start-i+self.window_size:end-i+self.window_size] = scores.squeeze(1)

        # Apply softmax on the last dimension to normalize the scores
        attention = torch.softmax(attention_scores, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhql,nhld->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        
        return self.fc_out(out)


# In[3]:


class TransformerClassifierWindow(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_classes, max_seq_len, window_size):
        super(TransformerClassifierWindow, self).__init__()
        self.feature_to_embed = nn.Linear(feature_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.sliding_window_attention = SlidingWindowAttention(embed_dim, window_size)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        x = self.feature_to_embed(x)  # [batch_size, seq_len, embed_dim]
        x += self.position_embedding[:, :x.size(1), :]  # Add position embedding
        x = self.sliding_window_attention(x)
        x = x.mean(dim=1)  # Pooling over the sequence dimension
        out = self.classifier(x)
        return out


# In[10]:


# class TransformerVAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, z_dim, num_classes, max_length):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, hidden_dim)
#         self.position_embedding = nn.Embedding(max_length, hidden_dim)

#         # Transformer encoder
#         self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
#         self.fc_mean = nn.Linear(hidden_dim, z_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, z_dim)

#         # Transformer decoder
#         self.transformer_decoder = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
#         self.fc_out = nn.Linear(hidden_dim, input_dim)

#         # Classification head
#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def encode(self, x):
#         embedded = self.embedding(x) + self.position_embedding(torch.arange(x.size(1), device=x.device))
#         encoded = self.transformer_encoder(embedded)
#         mean = self.fc_mean(encoded.mean(dim=1))
#         logvar = self.fc_logvar(encoded.mean(dim=1))
#         logits = self.classifier(encoded.mean(dim=1))
#         return mean, logvar, logits

#     def reparameterize(self, mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def decode(self, z):
#         decoded = self.transformer_decoder(z.unsqueeze(1))
#         return torch.sigmoid(self.fc_out(decoded))

#     def forward(self, x):
#         mean, logvar, logits = self.encode(x)
#         z = self.reparameterize(mean, logvar)
#         reconstructed = self.decode(z)
#         return reconstructed, mean, logvar, logits

# # Model parameters
# # input_dim = 1000  # Vocabulary size
# # hidden_dim = 512
# # z_dim = 20
# # num_classes = 10
# # max_length = 50

# # Initialize the model
# # model = TransformerVAE(input_dim, hidden_dim, z_dim, num_classes, max_length)

# # Example forward pass with random data
# # x = torch.randint(0, input_dim, (32, max_length))  # Example batch of sequences
# # reconstructed, mean, logvar, logits = model(x)


# In[11]:


class CustomDataset(Dataset):
    def __init__(self, feature_matrix, labels):
        self.features = torch.tensor(feature_matrix, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# In[7]:


class CustomDatasetSparse(Dataset):
    def __init__(self, feature_matrix, labels):
        self.features = torch.tensor(feature_matrix, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# In[1]:


class struc_var:
    def __init__(self, 
                 id, 
                 ref_name, 
                 sv_type, 
                 sv_pos, 
                 sv_stop, 
                 length, 
                 gt, 
                 wrong_len,
                 ref_len,
                 alt_len,
                 sample,
                 ac,
                 af=0):
        self.idx = str(sample) + "_" + str(id)
        self.ref_name = ref_name
        self.sv_pos = sv_pos
        self.sv_stop = sv_stop
        self.sv_type = sv_type
        self.length = length
        self.gt = gt
        self.wrong_len = wrong_len
        self.ref_len = ref_len
        self.alt_len = alt_len
        self.sample = sample
        self.id = id
        self.ac = ac
        self.af = af
        self.valid = None
        self.ttmars_valid = None

    def print_info(self):
        print(self.idx, self.ref_name, self.sv_pos, self.sv_stop, self.sv_type, self.length, self.gt, self.ac, self.af, self.valid, self.ttmars_valid)


# In[ ]:





# In[13]:


# Testing function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_seq, labels in test_loader:
            # put data to GPU
            input_seq = input_seq.cuda()
            labels = labels.cuda()
            
            outputs = model(input_seq)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# In[2]:


# filter to select sv

def filters(sv, sv_type, if_pass_only, sv_len):
    #type filter
    if sv_type not in valid_types:
        return True
    #PASS filter
    if if_pass_only:
        if 'PASS' not in sv.filter.keys():
            return True
    chr_name = sv.chrom
    #chr filter
    if chr_name not in chr_list:
        return True
    #len filter
    if sv_len < min_len or sv_len > max_len:
        return True
    return False


# In[2]:


# store input bed file into a list
def input_bed(bed_file):
    f = open(bed_file, 'r')
    lines = []
    for line in f:
        # print(line.split())
        lines.append(line.split())
    # close the file
    f.close()

    return lines

###########################################################################s
# get read depth

# get coverage of an entire interval
def get_cov_int(bam, ref_name, start, end):
    #note: bam.count_coverage will move the iterator to the end of end
    sv_cov = bam.count_coverage(ref_name, start, end)
    sv_cov_mat = torch.tensor(sv_cov)
    sv_cov_linear = torch.sum(sv_cov_mat, 0)

    return sv_cov_linear

#get depth for a given pos, may not be useful
def get_depth(ref_name, ref_pos, bam_file):
    pos_str = ref_name + ':' + str(int(ref_pos) - 1) + '-' + str(ref_pos)
    res = pysam.depth("-r", pos_str, bam_file)
    if res=='':
        return 0
    start = 0
    end = len(res) - 1
    for i in range(len(res) - 1, -1, -1):
        if res[i] == '\t':
            start = i + 1
            break
    return int(res[start:end])

#get read depth as an array for given positions
def get_rd(bam, sv):
    chr_name = sv.ref_name
    sv_start = sv.sv_pos
    sv_end = sv.sv_stop
    
    ref_name = chr_name
    ref_start = sv_start
    ref_end = sv_end

    #note: bam.count_coverage will move the iterator to the end of ref_end
    sv_cov = bam.count_coverage(ref_name, ref_start, ref_end)
    sv_cov_mat = np.array(sv_cov)
    sv_cov_linear = sv_cov_mat.sum(axis=0)

    return sv_cov_linear

############################################################################
## get sc count

#get the number of sc reads of a given interval
def get_sc_ctr(ref, start, end, bam):
    sc_ratio = 0.02
    sc_ctr = 0

    for read in bam.fetch(ref, start, end):
        cigar = read.cigartuples
        #not many reads are bad
        if not cigar:
            continue

        #discard if hard clipped
        try:
            if cigar[0][0] == 5:
                continue
            elif cigar[-1][0] == 5:
                continue
        except:
            continue

        read_len = read.infer_query_length()
        cur_sc_base = 0
        if cigar[0][0] == 4:
            cur_sc_base += cigar[0][1]
        if cigar[-1][0] == 4:
            cur_sc_base += cigar[-1][1]

        if cur_sc_base / read_len >= sc_ratio:
            sc_ctr += 1

    return sc_ctr

#get sc reads ctr
def get_sc(bam, ref_name, ref_start, ref_end):
    sc_stride = 25
    sv_sc = np.zeros(shape = (ref_end - ref_start))

    for cur_start in range(ref_start, ref_end, sc_stride):
        cur_sc_ctr = get_sc_ctr(ref_name, cur_start, cur_start + sc_stride, bam)

        for i in range(cur_start - ref_start, min(cur_start + sc_stride - ref_start, ref_end - ref_start)):
            sv_sc[i] = int(cur_sc_ctr)
            
    sv_sc = sv_sc.reshape(sv_sc.shape[0])
    sv_sc = torch.tensor(sv_sc)

    return sv_sc

###########################################################################
#return a simulated cn with a random locus and length
def simu_fp_cn(chr_len, min_len, max_len):
    rand_chr = random.randint(0, 21)
    rand_len = random.randint(min_len, max_len)
    # rand_len = int(rand_len // 1000 * 1000)
    
    chr_len = chr_len[rand_chr]
    rand_start_raw = random.randint(100 * rand_len, chr_len - 100 * rand_len)
    rand_start = round(rand_start_raw / 100) * 100
    
    return ['chr' + str(rand_chr + 1), str(rand_start), str(rand_start + rand_len)]


#format: ['chr1', '108500', '113000', '3.7', '']
def simu_fp_cn_draft(chr_len, min_len, max_len, all_copy_int, sample):
#     rand_chr = random.randint(0, 21)
    rand_chr = 0
    rand_len_raw = random.randint(min_len, max_len)
    rand_len = int(rand_len_raw // 1000 * 1000)
    
#     chr_len = chr_len[rand_chr]
    chr_len = 248000000
    rand_start_raw = random.randint(5 * rand_len, chr_len - 5 * rand_len)
    rand_start = round(rand_start_raw / 500) * 500
    
    while True:
        if check_ol(rand_start, rand_start + rand_len, all_copy_int):
            all_copy_int.append((rand_start, rand_start + rand_len))
            break
        
        rand_len_raw = random.randint(min_len, max_len)
        rand_len = int(rand_len_raw // 1000 * 1000)

#         chr_len = chr_len[rand_chr]
        rand_start_raw = random.randint(5 * rand_len, chr_len - 5 * rand_len)
        rand_start = round(rand_start_raw / 500) * 500      
    
    #test
#     print(['chr' + str(rand_chr + 1), str(rand_start), str(rand_start + rand_len), '1', ''])
    
    return ['chr' + str(rand_chr + 1), str(rand_start), str(rand_start + rand_len), '2', sample]

#embedding
###############################################
#get input seq on ref from vcf
def get_seq(sv, ref_fasta_file):
    chr_name = sv.ref_name
    sv_start = sv.sv_pos
    sv_end = sv.sv_stop
    ref_rec = ref_fasta_file.fetch(chr_name, sv_start, sv_end)
    return ref_rec

#encode function
def encode_kmer(k: int, kmer: str, table: dict) -> int:
    #order: A G C T <=> 0 1 2 3
    #min value: 0
    #max value: 4^k - 1
    if len(kmer) != k:
        return -1
    
    encode = 0
    for i in range(0, k):
        if kmer[i].upper() not in table:
            return -1
        encode += table[kmer[i].upper()] * (4**(k-1-i))
    
    return encode

#encode each seq
def encode_seq(input_seq, table, k):
    #embedding results
    embd_res = np.zeros(4**k, dtype=np.uintc)
    
    target_str = input_seq
    bad_kmer_ctr = 0
    for i in range(0, len(target_str) - k + 1):
        cur_kmer = target_str[i:i+k]
        encode = encode_kmer(k, cur_kmer, table)
        if encode != -1:
            embd_res[encode] += 1
        else:
            bad_kmer_ctr += 1
            
    return embd_res
    
###############################################

#simulate read depth
def simu_rd(same_len, lam, len_lb, len_up, avg_depth, depth_multi, chr_len):
    cur_start = 0
    cur_end = 0
    simu_res = []
    if same_len:
        length = 500
    while True:
        inter_arr_time = random.expovariate(lam)
        cur_pos = cur_end + int(inter_arr_time)
        if not same_len:
            cur_end = cur_pos + random.randint(len_lb, len_up)
        else:
            cur_end = cur_pos + length
        if cur_end >= chr_len:
            break
        event_depth = int(avg_depth * random.choice(depth_multi))
        simu_res.append([cur_pos, cur_end-1, event_depth])
    return simu_res

def simu_false_rd(same_len, lam, avg_depth, depth_multi, chr_len):
    cur_start = 0
    cur_end = 0
    simu_res = []
    if same_len:
        length = 500
    while True:
        inter_arr_time = random.expovariate(lam)
        cur_pos = cur_end + int(inter_arr_time)
        if not same_len:
            cur_end = cur_pos + random.randint(len_lb, len_up)
        else:
            cur_end = cur_pos + length
        if cur_end >= chr_len:
            break
#         event_depth = int(avg_depth * random.choice(depth_multi))
        #false event
        event_depth = int(avg_depth * 1)
        simu_res.append([cur_pos, cur_end-1, event_depth])
    return simu_res

def int_list_2_arr_list(list_of_int):
    for i in range(len(list_of_int)):
        list_of_int[i] = np.array([list_of_int[i]], dtype = float)

###############################################


#soft clipping counting
###############################################

###############################################

###########################################################################
#convert copy number to ordinal numbers (1, 1.5, 2, ...)
def covert_simu_cn(cn_in):
    cn = cn_in / 2
    #find the cloest value in 1, 1.5, ..., > 4
    if cn >= 4:
        return 4
    no_of_half = cn / 0.5
    return round(no_of_half) * 0.5
###########################################################################
#check if simu cnv overlapping with previous calls
def check_ol(start, end, pre_int_list):
    flank = 20000
    for cur_start, cur_end in pre_int_list:
        s = cur_start - flank
        e = cur_end + flank
        
        if end >= s and start <= e:
            return True
    
    return False

# to categorical data
def cate(y):
    cat = []
    for pt in y:
        temp = [0] * 7
        temp[int(pt/0.5) - 2] = 1
        cat.append(temp)
    return cat
    



# In[3]:


def overlap(sv1, sv2, ol_interval):
    x1, y1 = sv1.sv_pos, sv1.sv_stop
    x2, y2 = sv2.sv_pos - ol_interval, sv2.sv_stop + ol_interval

    if x1 <= y2 and y1 >= x2: return True
    else: return False
    


# In[2]:


def min_max_normalize(feature):
    v_min, v_max = feature.min(), feature.max()
    new_min, new_max = 0, 1

    res = (feature - v_min)/(v_max - v_min)*(new_max - new_min) + new_min

    return res


# In[ ]:


def get_features(x, y, input_list, label, bam, reversed):
    
    for call in input_list:
        cov = get_cov_int(bam, call[0], int(call[1]) - flanking, int(call[2]) + flanking)
        sc = get_sc(bam, call[0], int(call[1]) - flanking, int(call[2]) + flanking)
        kmer_emb = get_kmer_emb(call[0], int(call[1]) - flanking, int(call[2]) + flanking)
        insert_size = get_insert(bam, call[0], int(call[1]) - flanking, int(call[2]) + flanking)

        # filter crazy regions
        if cov.max() > cov_ub: continue

        if reversed:
            cov = torch.flip(cov, [0])
            sc = torch.flip(sc, [0])
            kmer_emb = torch.flip(kmer_emb, [0])
            insert_size = torch.flip(insert_size, [0])

        # cov = min_max_normalize(cov)
        # sc = min_max_normalize(sc)
        
        feature = torch.stack((cov, sc, kmer_emb, insert_size), dim = 0) 
        # feature = torch.stack((kmer_emb, insert_size), dim = 0) 
        
        padding_size = max_len + 2 * flanking - cov.size(0)
        feature_padded = F.pad(feature, (0, padding_size), 'constant', 0)
        #truncate cov_padded
        # feature_padded = torch.clamp(feature_padded, max = cov_ub)
        feature_padded = feature_padded.permute(1, 0)

        x.append(feature_padded.tolist())
        y.append(label)

    return


# In[ ]:


def get_features_with_key(x, y, keys, rec, label, bam, reversed, key):
    sample, ref, start, end = rec.sample, rec.ref_name, rec.sv_pos, rec.sv_stop
    
    cov = get_cov_int(bam, ref, int(start) - flanking, int(end) + flanking)
    sc = get_sc(bam, ref, int(start) - flanking, int(end) + flanking)
    kmer_emb = get_kmer_emb(call[0], int(call[1]) - flanking, int(call[2]) + flanking)
    insert_size = get_insert(bam, call[0], int(call[1]) - flanking, int(call[2]) + flanking)
    
    # filter crazy regions
    if cov.max() > cov_ub: return

    if reversed:
        cov = torch.flip(cov, [0])
        sc = torch.flip(sc, [0])
        kmer_emb = torch.flip(kmer_emb, [0])
        insert_size = torch.flip(insert_size, [0])

    # cov = min_max_normalize(cov)
    # sc = min_max_normalize(sc)
    
    feature = torch.stack((cov, sc, kmer_emb, insert_size), dim = 0) 
    
    padding_size = max_len + 2 * flanking - cov.size(0)
    feature_padded = F.pad(feature, (0, padding_size), 'constant', 0)
    #truncate cov_padded
    # feature_padded = torch.clamp(feature_padded, max = cov_ub)
    feature_padded = feature_padded.permute(1, 0)

    x.append(feature_padded.tolist())
    y.append(label)

    keys.append(key)


# In[16]:


def get_word2vec_emb(kmer_emb_int, Word2Vec_model):

    kmer_emb = []
    for kmer_int in kmer_emb_int:
        int_value = int(kmer_int)
        if int_value in kmer_reverse_mapping:
            kmer = kmer_reverse_mapping[int_value]
            if kmer in Word2Vec_model.wv:
                kmer_vector = Word2Vec_model.wv[kmer]
                kmer_emb.append(list(kmer_vector))
            else:
                kmer_emb.append([0] * kmer_emb_size)
        else:
            kmer_emb.append([0] * kmer_emb_size)

    return torch.tensor(kmer_emb)


# In[ ]:


def get_saved_features(x_file, y_file, x, y, reversed, Word2Vec_model):
    with open(x_file, 'r') as file:
        cov, sc, kmer_emb, insert_size = None, None, None, None
        
        for count, line in enumerate(file):
            if count % no_features == 0: 
                cov = torch.tensor([int(float(item)) for item in line.strip().split(' ')])
            elif count % no_features == 1: 
                sc = torch.tensor([int(float(item)) for item in line.strip().split(' ')])
            elif count % no_features == 2: 
                kmer_emb_int = torch.tensor([int(float(item)) for item in line.strip().split(' ')])
            elif count % no_features == 3: 
                insert_size = torch.tensor([int(float(item)) for item in line.strip().split(' ')])

                if cov.max() > cov_ub: continue

                if reversed:
                    cov = torch.flip(cov, [0])
                    sc = torch.flip(sc, [0])
                    kmer_emb_int = torch.flip(kmer_emb_int, [0])
                    insert_size = torch.flip(insert_size, [0])

                assert cov.size(dim=0) == sc.size(dim=0)
                assert cov.size(dim=0) == kmer_emb_int.size(dim=0)
                assert cov.size(dim=0) == insert_size.size(dim=0)

                kmer_emb = get_word2vec_emb(kmer_emb_int, Word2Vec_model)

                cov = cov.view(-1, 1)
                sc = sc.view(-1, 1)
                insert_size = insert_size.view(-1, 1)

                cov = cov.to(dtype=torch.float) / cov_scaler
                sc = sc.to(dtype=torch.float) / sc_scaler
                insert_size = insert_size.to(dtype=torch.float) / insert_scaler
                kmer_emb = kmer_emb.to(dtype=torch.float) / emb_scaler

                # Concatenate the tensors
                feature = torch.cat((cov, sc, insert_size, kmer_emb), dim=1)
                # feature = torch.stack((cov, sc, insert_size, kmer_emb), dim = 0) 
                # feature = torch.stack((kmer_emb, insert_size), dim = 0) 
                
                padding_size = max_len + 2 * flanking - cov.size(0)
                # feature_padded = F.pad(feature, (0, padding_size), 'constant', 0)
                feature_padded = F.pad(feature, (0, 0, 0, padding_size), 'constant', 0)
                
                #truncate cov_padded
                # feature_padded = torch.clamp(feature_padded, max = cov_ub)
                # feature_padded = feature_padded.permute(1, 0)
        
                x.append(feature_padded.tolist())
    
    with open(y_file, 'r') as file:
        for count, line in enumerate(file):
            y.append(int(line.strip().split(' ')[0]))


# In[ ]:


def get_saved_features_with_keys(x, y, keys, rec, label, reversed, key, Word2Vec_model, feature_list):
    sample, ref, start, end = rec.sample, rec.ref_name, rec.sv_pos, rec.sv_stop 
    
    cov, sc, kmer_emb_int, insert_size = feature_list

    if cov.max() > cov_ub: return

    if reversed:
        cov = torch.flip(cov, [0])
        sc = torch.flip(sc, [0])
        kmer_emb_int = torch.flip(kmer_emb_int, [0])
        insert_size = torch.flip(insert_size, [0])

    assert cov.size(dim=0) == sc.size(dim=0)
    assert cov.size(dim=0) == kmer_emb_int.size(dim=0)
    assert cov.size(dim=0) == insert_size.size(dim=0)

    kmer_emb = get_word2vec_emb(kmer_emb_int, Word2Vec_model)

    cov = cov.view(-1, 1)
    sc = sc.view(-1, 1)
    insert_size = insert_size.view(-1, 1)

    cov = cov.to(dtype=torch.float) / cov_scaler
    sc = sc.to(dtype=torch.float) / sc_scaler
    insert_size = insert_size.to(dtype=torch.float) / insert_scaler
    kmer_emb = kmer_emb.to(dtype=torch.float) / emb_scaler

    # Concatenate the tensors
    feature = torch.cat((cov, sc, insert_size, kmer_emb), dim=1)
    # feature = torch.stack((cov, sc, insert_size, kmer_emb), dim = 0) 
    # feature = torch.stack((kmer_emb, insert_size), dim = 0) 
    
    padding_size = max_len + 2 * flanking - cov.size(0)
    # feature_padded = F.pad(feature, (0, padding_size), 'constant', 0)
    feature_padded = F.pad(feature, (0, 0, 0, padding_size), 'constant', 0)
    
    #truncate cov_padded
    # feature_padded = torch.clamp(feature_padded, max = cov_ub)
    # feature_padded = feature_padded.permute(1, 0)

    x.append(feature_padded.tolist())
    y.append(label)
    keys.append(key)


# In[ ]:


# get insert size feature: to implement

def get_insert(bam, ref_name, ref_start, ref_end):
    insert_stride = 25
    insert_size = np.zeros(shape = (ref_end - ref_start))

    for cur_start in range(ref_start, ref_end, insert_stride):
        cur_insert_size = get_insert_ctr(ref_name, cur_start, cur_start + insert_stride, bam)

        for i in range(cur_start - ref_start, min(cur_start + insert_stride - ref_start, ref_end - ref_start)):
            insert_size[i] = int(cur_insert_size)
            
    insert_size = insert_size.reshape(insert_size.shape[0])
    insert_size = torch.tensor(insert_size)

    return insert_size

def  get_insert_ctr(ref, start, end, bam):
    insert_sum, read_ctr = 0, 0
    for read in bam.fetch(ref, start, end):
        if read.is_paired:
            # if read.is_read1:
            insert_size = abs(read.template_length)  # Use abs to ensure the size is positive
            insert_sum += insert_size
            read_ctr += 1

    if read_ctr != 0: 
        return int(insert_sum / read_ctr)
    else:
        return 0


# In[ ]:


# get local 3-mer embedding features

def get_kmer_emb(ref_name, ref_start, ref_end):
    k = 3
    kmer_emb = np.zeros(shape = (ref_end - ref_start))

    for cur in range(ref_start, ref_end):
        kmer = extract_kmer_from_reference(ref_name, cur - 1, k)
        emb = emb_kmer(kmer)
        kmer_emb[cur - ref_start] = emb

    kmer_emb = kmer_emb.reshape(kmer_emb.shape[0])
    kmer_emb = torch.tensor(kmer_emb)

    return kmer_emb

def emb_kmer(kmer):
    k = 3
    kmer = kmer.upper()

    if kmer not in kmer_mapping: return 0
    else: return kmer_mapping[kmer]
    
def extract_kmer_from_reference(chromosome, position, k=3):
    # Open the reference genome FASTA file
    with pysam.FastaFile(fasta_file_path) as fasta_file:
        kmer = fasta_file.fetch(chromosome, position, position + k)
        return kmer


# In[ ]:


# prediction loop

def predict(model, predictions, x, y, batch_size, shuffle):
    dataset = CustomDataset(x, y)

    pred_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Prediction loop
    with torch.no_grad():
        for input_ids, _ in pred_loader:  # DataLoader: (input_ids, labels)
            input_ids = input_ids.cuda()
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.tolist())


# In[ ]:


def get_sv_len_from_feature(line_list):
    padded_len = len(line_list)
    zero_ctr = 0

    for i in range(len(line_list) - 1, -1, -1):
        if line_list[i] == '0.0': zero_ctr += 1
        else: break

    return padded_len - zero_ctr


def index_vcf(vcf_file, cur_valid_types):
    sv_dict = dict()
    
    ###################################
    f = pysam.VariantFile(vcf_file,'r')
    
    for count, rec in enumerate(f.fetch()):
        #get sv_type
        try:
            sv_type = rec.info['SVTYPE']
        except:
            print("invalid sv type info")
            continue
    
        #get sv length
        if sv_type == 'INV':
            sv_len = abs(rec.stop - rec.pos + 1)
        else:
            try:
                sv_len = rec.info['SVLEN'][0]
            except:
                try:
                    sv_len = rec.info['SVLEN']
                except:
                    sv_len = abs(rec.stop - rec.pos + 1)

        if sv_type not in cur_valid_types: continue

        if sv_type == "INS": cur_min_len, cur_max_len = 50, 500
        elif sv_type == "DEL": cur_min_len, cur_max_len = 400, 2000
        elif sv_type == "DUP": cur_min_len, cur_max_len = 400, 2000
        elif sv_type == "INV": cur_min_len, cur_max_len = 50, 2000
        
        if sv_len < cur_min_len or sv_len > cur_max_len: continue
            
        if filters(rec, sv_type, True, sv_len):
            continue
    
        sv_gt = None
        
        ref_len = len(rec.ref)
        alt_len = len(rec.alts[0])
        
        sv_dict[str(sample) + "_" + str(count)] = struc_var(count, rec.chrom, sv_type, rec.pos, rec.stop, 
                                                            sv_len, sv_gt, False, ref_len, alt_len, sample,
                                                            ac = 1)
        
    f.close()
    
    return sv_dict

def output_bed(sv_dict, bed_file):
    with open(bed_file, 'w') as file:
        # open file in read mode
        f = open(ttmars_res, 'r')
        
        for sv in sv_dict:
            file.write(sv.ref_name + '\t')
            file.write(sv.sv_pos + '\t')
            file.write(sv.sv_stop + '\t' + '\n')
        
        # close the file
        f.close()
