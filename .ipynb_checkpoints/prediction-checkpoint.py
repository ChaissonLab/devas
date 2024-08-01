################################################################
################################################################

import pysam
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader
from gensim.models import Word2Vec

import argparse
import sys
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# use fn and class from utils.

from utils import *

################################################################
################################################################

def index_vcf(sv_dict, samples, vcf_file_part1, vcf_file_part2, cur_valid_types):
    for sample in samples:
        print(sample)
        
        ###################################
        # vcf_file = f"/scratch1/jianzhiy/dl_cnv/run_delly/run_with_ins/{sample}/delly.vcf"
        vcf_file = vcf_file_part1 + sample + vcf_file_part2
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
                        #print("invalid sv length info")
        #         try:
        #             sv_len = rec.info['SVLEN'][0]
        #         except:
        #             sv_len = rec.info['SVLEN']
            #handle del length > 0:
            # if sv_type == 'DEL':
            #     sv_len = -abs(sv_len)
    
            if sv_type not in cur_valid_types: continue

            if sv_type == "INS": cur_min_len, cur_max_len = 50, 500
            elif sv_type == "DEL": cur_min_len, cur_max_len = 400, 2000
            elif sv_type == "DUP": cur_min_len, cur_max_len = 400, 2000
            elif sv_type == "INV": cur_min_len, cur_max_len = 50, 2000
            
            if sv_len < cur_min_len or sv_len > cur_max_len: continue
                
            if filters(rec, sv_type, True, sv_len):
                continue
        
            sv_gt = None
            
        #     if len(rec.samples.values()) != 1:
        #         raise Exception("Wrong number of sample genotype(s)")
        #     gts = [s['GT'] for s in rec.samples.values()] 
            ref_len = len(rec.ref)
            alt_len = len(rec.alts[0])
            
            sv_dict[str(sample) + "_" + str(count)] = struc_var(count, rec.chrom, sv_type, rec.pos, rec.stop, 
                                                                sv_len, sv_gt, False, ref_len, alt_len, sample,
                                                                ac = 1)
            
        f.close()
    
    
################################################################
################################################################
# Index and load model

sv_dict = dict()
index_vcf(sv_dict, samples, vcf_file_part1, vcf_file_part2, cur_valid_types)

#DEL
# Model Parameters
version = "19_3"

BinaryClassificationTransformer11117Norm2
d_model = 32
nhead = 4
num_layers = 2
num_classes = 2
dim_feedforward = 256
local_attn_ctx = 10
blocksize = 16

# load trained model and predict

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

del_model = BinaryClassificationTransformer11117Norm2(d_model=d_model, 
                                                        nhead=nhead, 
                                                        attn_mode='strided', 
                                                        local_attn_ctx=local_attn_ctx, 
                                                        blocksize=blocksize, 
                                                        num_layers=num_layers, 
                                                        dim_feedforward=dim_feedforward)

# Load the saved state dict
del_model.load_state_dict(torch.load(f'../models/transformer_classifier_v{version}.pth'))
del_model.to(device)
del_model.eval()

# Load testing data

# DEL
# get features from saved

x_del, y_del = [], []
Word2Vec_model_name = "DEL_delly_400_2000_30samples_3mers_truth__word2vec"
Word2Vec_model = Word2Vec.load(f"{Word2Vec_model_name}.model")

for i in range(len(testing_samples)):
    # if i % 2 == 1: continue

    sample1 = testing_samples[i]
    # sample2 = samples[i + 1]
    
    #test
    print(f"{sample1}_")

    x_feature_file = f"/scratch1/jianzhiy/dl_cnv/training_features/delly/x_{sample1}_cov_sc_del_kmer_insert.txt"
    y_feature_file = f"/scratch1/jianzhiy/dl_cnv/training_features/delly/y_{sample1}_cov_sc_del_kmer_insert.txt"

    get_saved_features(x_feature_file, y_feature_file, x_del, y_del, False, Word2Vec_model)

################################################################
################################################################
# Prediction

# DEL
predictions_del = []
predict(del_model, predictions_del, x_del, y_del, 32, False)
