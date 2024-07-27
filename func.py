import numpy as np
import random


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


#read depth
###############################################
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

###########################################################################
#return a simulated cn with a random locus and length
#format: ['chr1', '108500', '113000', '3.7', '']
def simu_fp_cn(chr_len, min_len, max_len, all_copy_int, sample):
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

#get the number of sc reads of a given interval
def get_sc_ctr(ref, start, end, bam):
    sc_ratio = 0.2
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

