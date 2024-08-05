# %run '../utils_ins.ipynb'

import pysam
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

import argparse
import sys
import os
# Add the parent directory to sys.path to allow importing
parent_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir_path)
# sys.path.append('/project/mchaisso_100/cmb-16/quentin/workspace/notebooks/')

###################################
# parameters

from utils import *

cur_min_len = 400
cur_max_len = 2000

cur_valid_types = ['DEL']

# true_bed = true_bed_part1 + sample + true_bed_part2
# true_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly_ins/{sample}/ttmars_combined_true_ins_50_500.bed"
truth_bed_part1 = "/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/"
truth_bed_part2 = "/ttmars_combined_true_" + cur_valid_types[0] + f"_{str(cur_min_len)}_{str(cur_max_len)}.bed"

# false_bed = false_bed_part1 + sample + false_bed_part2
# false_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly_ins/{sample}/ttmars_combined_false_ins_50_500.bed"
false_bed_part1 = "/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly/"
false_bed_part2 = "/ttmars_combined_false_" + cur_valid_types[0] + f"_{str(cur_min_len)}_{str(cur_max_len)}.bed"

# output x and y file: x_file = x_file_part1 + sample_name + x_file_part2
# x_file = f"/scratch1/jianzhiy/dl_cnv/training_features/delly/x_{sample}_cov_sc_del_kmer_insert_ins_50_500.txt"
x_file_part1 = "/scratch1/jianzhiy/dl_cnv/training_features/delly/x_"
x_file_part2 = f"_cov_sc_kmer_insert_{cur_valid_types[0]}_{str(cur_min_len)}_{str(cur_max_len)}.txt"

y_file_part1 = "/scratch1/jianzhiy/dl_cnv/training_features/delly/y_"
y_file_part2 = f"_cov_sc_kmer_insert_{cur_valid_types[0]}_{str(cur_min_len)}_{str(cur_max_len)}.txt"

###################################

# Create the parser
parser = argparse.ArgumentParser(description="Process some samples.")
# Add arguments
parser.add_argument('sample', type=str)
# Parse the arguments
args = parser.parse_args()

print("Sample received:", args.sample)
sample = str(args.sample)
x, y = [], []

#test
print(sample, len(x))



# input SV coordinate
# truth_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly_ins/{sample}/ttmars_combined_true_ins_50_500.bed"
truth_bed = truth_bed_part1 + sample + truth_bed_part2
truth_list = input_bed(truth_bed)

# false_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly_ins/{sample}/ttmars_combined_false_ins_50_500.bed"
false_bed = false_bed_part1 + sample + false_bed_part2
false_list = input_bed(false_bed)

# Extract features:
bam_file = f"/scratch1/jianzhiy/data/illumina/1kg_related/hprc_overlap/{sample}/{sample}.bam"
bam = pysam.AlignmentFile(bam_file, "rb")

get_features(x, y, truth_list, 1, bam, False)
get_features(x, y, false_list, 0, bam, False)

# save training features
# x_file = f"/scratch1/jianzhiy/dl_cnv/training_features/delly/x_{sample}_cov_sc_del_kmer_insert_ins_50_500.txt"
x_file = x_file_part1 + sample + x_file_part2
# y_file = f"/scratch1/jianzhiy/dl_cnv/training_features/delly/y_{sample}_cov_sc_del_kmer_insert_ins_50_500.txt"
y_file = y_file_part1 + sample + y_file_part2

with open(x_file, "w") as file_x, open(y_file, "w") as file_y:
    for i in range(len(x)):
        cur_x, cur_y = x[i], y[i]

        for features in cur_x: file_x.write(f"{features[0]} ")
        file_x.write("\n")
        for features in cur_x: file_x.write(f"{features[1]} ")
        file_x.write("\n")

        for features in cur_x: file_x.write(f"{features[2]} ")
        file_x.write("\n")
        for features in cur_x: file_x.write(f"{features[3]} ")
        file_x.write("\n")

        file_y.write(f"{cur_y} ")
        file_y.write("\n")