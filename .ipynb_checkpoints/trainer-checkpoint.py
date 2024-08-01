################################################################
################################################################

# use fn and class from utils.

from utils import *

# Import

import pysam
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

# chech torch cuda status
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

################################################################
################################################################

samples = ['HG00438']

# output true and false sv bed coordinate files after filtering
# index sv

sv_dict = dict()
for sample in samples:
    vcf_file = f"/scratch1/jianzhiy/dl_cnv/run_delly/{sample}/delly.vcf"
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

        # if sv_type == 'INS': print(sv_len)
            
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

# output true and false sv bed coordinate files after filtering

# output bed file with ttmars results

#DEL, INS, DUP counter
true_types_counter = [0, 0, 0]
false_types_counter = [0, 0, 0]

# true calls
for sample in samples:
    true_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_true.bed"
    with open(true_bed, 'w') as file:
        
        ttmars_res = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_res.txt"
    
        # open file in read mode
        f = open(ttmars_res, 'r')
        
        for line in f:
            # print(line.split())
            line_list = line.split()
            idx = str(sample) + '_' + str(line_list[0])

            # if a true call passed the filter
            if idx in sv_dict and str(line_list[3]) == 'True':
                file.write(str(line_list[4]) + '\t')
                file.write(str(line_list[5]) + '\t')
                file.write(str(line_list[6]) + '\t' + '\n')

                if sv_dict[idx].sv_type == 'DEL': true_types_counter[0] += 1
                elif sv_dict[idx].sv_type == 'INS': true_types_counter[1] += 1
                elif sv_dict[idx].sv_type == 'DUP': true_types_counter[2] += 1
        
        # close the file
        f.close()


# false calls
for sample in samples:
    false_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_false.bed"
    with open(false_bed, 'w') as file:
        
        ttmars_res = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_res.txt"
    
        # open file in read mode
        f = open(ttmars_res, 'r')
        
        for line in f:
            # print(line.split())
            line_list = line.split()
            idx = str(sample) + '_' + str(line_list[0])

            # if a true call passed the filter
            if idx in sv_dict and str(line_list[3]) == 'False':
                file.write(str(line_list[4]) + '\t')
                file.write(str(line_list[5]) + '\t')
                file.write(str(line_list[6]) + '\t' + '\n')

                if sv_dict[idx].sv_type == 'DEL': false_types_counter[0] += 1
                elif sv_dict[idx].sv_type == 'INS': false_types_counter[1] += 1
                elif sv_dict[idx].sv_type == 'DUP': false_types_counter[2] += 1
        
        # close the file
        f.close()


################################################################
################################################################

x, y = [], []

for sample in samples:
    #test
    print(sample)
    
    # input SV coordinate
    truth_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_true.bed"
    truth_list = input_bed(truth_bed)

    false_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_false.bed"
    false_list = input_bed(false_bed)
    
    # # Simu false list
    # false_list = []
    # for _ in range(len(truth_list)):
    #     false_list.append(simu_fp_cn(chr_len, min_len, max_len))
    
    # Extract features: read depth
    bam_file = f"/scratch1/jianzhiy/data/illumina/1kg_related/hprc_overlap/{sample}/{sample}.bam"
    bam = pysam.AlignmentFile(bam_file, "rb")

    get_features(x, y, truth_list, 1, bam, False)
    get_features(x, y, false_list, 0, bam, False)

################################################################
################################################################

# Parameters for model
d_model = 8
nhead = 2
num_layers = 2
num_classes = 2
dim_feedforward = 16
local_attn_ctx = 10
blocksize = 16

batch_size = 16

################################################################
################################################################

# Split input train test datasets

dataset = CustomDataset(x, y)
# dataset = CustomDatasetSparse(x, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# to balance True False classes

num_positive_samples = 0
num_negative_samples = 0

for sample in samples:
    # input SV coordinate
    truth_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_true.bed"
    truth_list = input_bed(truth_bed)
    num_positive_samples += len(truth_list)

    false_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/{sample}/ttmars_combined_false.bed"
    false_list = input_bed(false_bed)
    num_negative_samples += len(false_list)

# Calculating weights
total_samples = num_positive_samples + num_negative_samples
weight_for_positive = total_samples / (2.0 * num_positive_samples)
weight_for_negative = total_samples / (2.0 * num_negative_samples)

class_weights = torch.tensor([weight_for_negative, weight_for_positive], dtype=torch.float)

# Move to cuda if available
if torch.cuda.is_available():
    print('cuda is here')
    class_weights = class_weights.cuda()

print(class_weights)

################################################################
################################################################

#Define models
model = BinaryClassificationTransformer11117(d_model=d_model, 
                                            nhead=nhead, 
                                            attn_mode='strided', 
                                            local_attn_ctx=local_attn_ctx, 
                                            blocksize=blocksize, 
                                            num_layers=num_layers, 
                                            dim_feedforward=dim_feedforward)
model.to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)


################################################################
################################################################
# Training loop

num_epochs = 100

tic = time.perf_counter()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_seq, labels in train_loader:
        # put data to GPU
        input_seq = input_seq.cuda()
        labels = labels.cuda()

        # print("input_seq ", input_seq.size())
        
        outputs = model(input_seq)
        
        # print("outputs", outputs.size(), labels.size())
        
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_accuracy = test_model(model, train_loader)
    test_accuracy = test_model(model, test_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

toc = time.perf_counter()
print(f"Time in {toc - tic:0.4f} seconds")


################################################################
################################################################
# Save Model

version = "8_7"
torch.save(model.state_dict(), f'../models/transformer_classifier_v{version}.pth')








