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

###################################
# parameters

from utils import *

def main():
    parser.add_argument('-t', '--type', type=str, help='Type of SV', required=True)
    parser.add_argument('-v', '--vcf_file', type=str, help='Input vcf file', required=True)
    parser.add_argument('-b', '--bam_file', type=str, help='Bam file', required=True)
    parser.add_argument('-o', '--output_file', type=str, help='Output feature file', required=True)
    parser.add_argument('-d', '--working_dir', type=str, help='Working directory', required=True)

    # Parse the arguments
    args = parser.parse_args()

    cur_valid_types = args.type
    vcf_file = args.type
    x_file = args.output_file
    working_dir = args.working_dir

    # Extract features:
    bam_file = args.bam
    bam = pysam.AlignmentFile(bam_file, "rb")
    
    cur_min_len = 400
    cur_max_len = 2000
    x, y = [], []

    sv_dict = index_vcf(vcf_file, cur_valid_types)
    
    # input SV coordinate
    # truth_bed = f"/scratch1/jianzhiy/dl_cnv/run_ttmars/vali_delly_ins/{sample}/ttmars_combined_true_ins_50_500.bed"
    truth_bed = truth_bed_part1 + sample + truth_bed_part2
    truth_list = input_bed(truth_bed)
    
    get_features(x, y, truth_list, 1, bam, False)
    
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



if __name__ == "__main__":
    main()