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
    
    bed_file = working_dir + "/sv_bed_file.bed"
    output_bed(sv_dict, bed_file)
    sv_list = input_bed(bed_file)
    
    get_features(x, y, sv_list, 1, bam, False)
    
    with open(x_file, "w") as file_x:
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

if __name__ == "__main__":
    main()