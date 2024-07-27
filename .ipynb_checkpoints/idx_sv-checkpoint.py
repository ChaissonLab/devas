#Import
#################################################################
import pysam

#CONST
#################################################################
#flags
seq_resolved = False
if_pass_only = True
gt_vali = False
if_hg38 = True


#chr names
chr_list = []
if if_hg38:
    chr_list = ["chr1", "chr2", "chr3", "chr4", "chr5",
                "chr6", "chr7", "chr8", "chr9", "chr10",
                "chr11", "chr12", "chr13", "chr14", "chr15",
                "chr16", "chr17", "chr18", "chr19", "chr20",
                "chr21", "chr22", "chrX"]
else:
    chr_list = ["1", "2", "3", "4", "5",
                "6", "7", "8", "9", "10",
                "11", "12", "13", "14", "15",
                "16", "17", "18", "19", "20",
                "21", "22", "X"]
#approximate length of chromosomes
chr_len = [250000000, 244000000, 199000000, 192000000, 182000000, 
            172000000, 160000000, 147000000, 142000000, 136000000, 
            136000000, 134000000, 116000000, 108000000, 103000000, 
            90400000, 83300000, 80400000, 59200000, 64500000, 
            48200000, 51400000, 157000000, 59400000]

#max/min length of allowed SV not DUP
memory_limit = 100000
memory_min = 10
#max length of allowed DUP
dup_memory_limit = 50000
dup_memory_min = 10

#valid types
valid_types = ['DEL', 'INS', 'INV', 'DUP:TANDEM', 'DUP']

#input vcf file
vcf_file = ""
#################################################################

#return False if not filtered
#first_filter: type, PASS, chr_name
def first_filter(sv, sv_type):
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
    return False

#second_filter: centromere, non-cov
def second_filter(sv):
    index = sv.idx
    ref_name = sv.ref_name
    sv_pos = sv.sv_pos
    sv_stop = sv.sv_stop

    if if_hg38:
        centro_start = int(dict_centromere[ref_name][0])
        centro_end = int(dict_centromere[ref_name][1])
    else:
        centro_start = int(dict_centromere['chr'+ref_name][0])
        centro_end = int(dict_centromere['chr'+ref_name][1])

    #centromere
    if (sv_pos > centro_start and sv_pos < centro_end) or (sv_stop > centro_start and sv_stop < centro_end):
        sv.is_sec_fil = True
        return True
        
    #non-cov
    list_to_check = [str(ref_name), str(sv_pos), str(sv_stop)]
    #if sv in high-depth regions or non-covered regions, skip
    if validate.check_exclude(list_to_check, exclude_assem1_non_cover, exclude_assem2_non_cover):
        sv.is_sec_fil = True
        return True
    
#third_filter: size
def third_filter(sv):
    #size
    if sv.sv_type not in ['DUP:TANDEM', 'DUP']:
        if abs(sv.length) < memory_min or abs(sv.length) > memory_limit:
            sv.is_third_fil = True
            return True
    else:
        if abs(sv.length) < dup_memory_min or abs(sv.length) > dup_memory_limit:
            sv.is_third_fil = True
            return True

#define class
class struc_var:
    def __init__(self, idx, ref_name, sv_type, sv_pos, sv_stop, length, gt):
        self.idx = idx
        self.ref_name = ref_name
        self.sv_pos = sv_pos
        self.sv_stop = sv_stop
        self.sv_type = sv_type
        self.length = length
        self.gt = gt
        #if the call is part of an aggregate SV
        self.is_agg = False
        #if second filtered out
        self.is_sec_fil = False
        self.is_third_fil = False
        
        self.query_name_hap1 = "NA"
        self.query_name_hap2 = "NA"
        
        self.ref_start_best_hap1 = -1
        self.ref_end_best_hap1 = -1
        self.query_start_best_hap1 = -1
        self.query_end_best_hap1 = -1
        
        self.ref_start_best_hap2 = -1
        self.ref_end_best_hap2 = -1
        self.query_start_best_hap2 = -1
        self.query_end_best_hap2 = -1
        
        self.analyzed_hap1 = False
        self.analyzed_hap2 = False
        
        self.len_query_hap1 = -1
        self.len_ref_hap1 = -1
        self.len_query_hap2 = -1
        self.len_ref_hap2 = -1
        
        self.score_before_hap1 = -1
        self.score_after_hap1 = -1
        self.score_before_hap2 = -1
        self.score_after_hap2 = -1
        
        self.neg_strand_hap1 = False
        self.neg_strand_hap2 = False
        
        self.ins_seq = ""
        self.if_seq_resolved = False
        
    def check_tp(self, rela_len, rela_score):
        result = True
        if self.sv_type in ['DEL', 'DUP', 'DUP:TANDEM']:
            if rela_score >= 0 and rela_score <= 2.5:
                if rela_len >= -0.05*rela_score + 0.8 and rela_len <= 0.05*rela_score + 1.2:
                    result = True
                else:
                    result = False
            elif rela_score > 2.5:
                if rela_len >= 0.675 and rela_len <= 1.325:
                    result = True
                else:
                    result = False
            else:
                result = False
        elif self.sv_type == 'INS':
            #not seq-resolved
            #if len(self.ins_seq) == 0:
            if not self.if_seq_resolved:
                if rela_len < 0.675 or rela_len > 1.325:
                    result = False
            #seq-resolved
            else:
                if rela_score >= 0 and rela_score <= 2.5:
                    if rela_len >= -0.05*rela_score + 0.8 and rela_len <= 0.05*rela_score + 1.2:
                        result = True
                    else:
                        result = False
                elif rela_score > 2.5:
                    if rela_len >= 0.675 and rela_len <= 1.325:
                        result = True
                    else:
                        result = False
                else:
                    result = False                
                
        elif self.sv_type == 'INV':
            if rela_score <= 0:
                result = False
        return result
    
    #TP when wrong length flag presents -- looser rules for TP
    def check_tp_wlen(self, rela_len, rela_score):
        result = True
        if self.sv_type in ['DEL', 'DUP', 'DUP:TANDEM']:
            if rela_score >= 0 and rela_score <= 2.5:
                if rela_len >= -0.05*rela_score + 0.6 and rela_len <= 0.05*rela_score + 1.4:
                    result = True
                else:
                    result = False
            elif rela_score > 2.5:
                if rela_len >= 0.475 and rela_len <= 1.525:
                    result = True
                else:
                    result = False
            else:
                result = False
        elif self.sv_type == 'INS':
            #not seq-resolved
            #if len(self.ins_seq) == 0:
            if not self.if_seq_resolved:
                if rela_len < 0.475 or rela_len > 1.525:
                    result = False
            #seq-resolved
            else:
                if rela_score >= 0 and rela_score <= 2.5:
                    if rela_len >= -0.05*rela_score + 0.6 and rela_len <= 0.05*rela_score + 1.4:
                        result = True
                    else:
                        result = False
                elif rela_score > 2.5:
                    if rela_len >= 0.475 and rela_len <= 1.525:
                        result = True
                    else:
                        result = False
                else:
                    result = False                
                
        elif self.sv_type == 'INV':
            if rela_score <= 0:
                result = False
        return result
        
    def print_info(self):
        print(self.idx, self.ref_name, self.sv_pos, self.sv_stop, self.sv_type, self.length, self.gt, self.is_agg, self.is_sec_fil, self.is_third_fil)
        
    def cal_rela_score(self, score_before, score_after):
        if score_before > -1 and score_before < 0:
            tmp_score_before = -1
            tmp_score_after = score_after + (tmp_score_before - score_before)
            return round((tmp_score_after - tmp_score_before) / abs(tmp_score_before), 2)
        
        elif score_before >= 0 and score_before < 1:
            tmp_score_before = 1
            tmp_score_after = score_after + (tmp_score_before - score_before)
            return round((tmp_score_after - tmp_score_before) / abs(tmp_score_before), 2)
        
        else:
            return round((score_after - score_before) / abs(score_before), 2)
        
    def cal_rela_len(self, query_len, ref_len):
        return round((query_len - ref_len) / self.length, 2)
        
    def get_vali_res(self):
        if (not self.analyzed_hap1) or (not self.analyzed_hap2):
            return -1
        
        if self.analyzed_hap1 and self.analyzed_hap2:
            rela_len_1 = self.cal_rela_len(self.len_query_hap1, self.len_ref_hap1)
            rela_len_2 = self.cal_rela_len(self.len_query_hap2, self.len_ref_hap2)
            
            rela_score_1 = self.cal_rela_score(self.score_before_hap1, self.score_after_hap1)
            rela_score_2 = self.cal_rela_score(self.score_before_hap2, self.score_after_hap2)
            
            if not wrong_len:
                res_hap1 = self.check_tp(rela_len_1, rela_score_1)
                res_hap2 = self.check_tp(rela_len_2, rela_score_2)
            else:
                res_hap1 = self.check_tp_wlen(rela_len_1, rela_score_1)
                res_hap2 = self.check_tp_wlen(rela_len_2, rela_score_2)
            
            gt_validate = False
            if args.gt_vali:
                if res_hap1 and res_hap2:
                    if self.gt == (1,1):
                        gt_validate = True
                elif res_hap1 or res_hap2:
                    if self.gt == (1,0) or self.gt == (0,1):
                        gt_validate = True
                
            if res_hap1 and res_hap2:
                if abs(rela_len_1 - 1) <= abs(rela_len_2 - 1):
                    return (res_hap1, rela_len_1, rela_score_1, gt_validate)
                else:
                    return (res_hap2, rela_len_2, rela_score_2, gt_validate)
            elif res_hap1:
                return (res_hap1, rela_len_1, rela_score_1, gt_validate)
            elif res_hap2:
                return (res_hap2, rela_len_2, rela_score_2, gt_validate)
            else:
                if abs(rela_len_1 - 1) <= abs(rela_len_2 - 1):
                    return (res_hap1, rela_len_1, rela_score_1, gt_validate)
                else:
                    return (res_hap2, rela_len_2, rela_score_2, gt_validate)
        
#index sv
def idx_sv(vcf_file):
    #index SVs
    f = pysam.VariantFile(vcf_file,'r')
    sv_list = []
    for count, rec in enumerate(f.fetch()):
        #get sv_type
        try:
            sv_type = rec.info['SVTYPE']
        except:
            print("invalid sv type info")
            continue

        if first_filter(rec, sv_type):
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
        #handle del length > 0:
        if sv_type == 'DEL':
            sv_len = -abs(sv_len)
        
        #skip small variants
        if abs(sv_len) < memory_min:
            continue

        #only taking the first sample genotype 
        if gt_vali:
            sv_gt = rec.samples[0]["GT"]
            #bad genotype
            if sv_gt not in [(1, 1), (1, 0), (0, 1)]:
                sv_gt = None
        else:
            sv_gt = None
        
        sv_list.append(struc_var(count, rec.chrom, sv_type, rec.pos, rec.stop, sv_len, sv_gt))   
        
        #add ins seq for seq-resolved insertion
        #no multi-allelic considered
        if (sv_type == 'INS') and seq_resolved:
            sv_list[len(sv_list)-1].ins_seq = rec.alts[0]
            sv_list[len(sv_list)-1].if_seq_resolved = True
        
    f.close()
    
    #index sv: second_filter: centromere, non-cov
    #third_filter: size
    for sv in sv_list:
#         second_filter(sv)
        third_filter(sv)

    return sv_list
        
#main function
def main():
    #index SVs
    f = pysam.VariantFile(vcf_file,'r')
    sv_list = []
    for count, rec in enumerate(f.fetch()):
        #get sv_type
        try:
            sv_type = rec.info['SVTYPE']
        except:
            print("invalid sv type info")
            continue

        if first_filter(rec, sv_type):
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
        #handle del length > 0:
        if sv_type == 'DEL':
            sv_len = -abs(sv_len)
            
        if abs(sv_len) < memory_min:
            continue

        #get gt
        #only taking the first sample genotype 
        if args.gt_vali:
            sv_gt = rec.samples[0]["GT"]
            #bad genotype
            if sv_gt not in [(1, 1), (1, 0), (0, 1)]:
                sv_gt = None
        else:
            sv_gt = None
        
        sv_list.append(struc_var(count, rec.chrom, sv_type, rec.pos, rec.stop, sv_len, sv_gt))   
        
        #add ins seq for seq-resolved insertion
        #no multi-allelic considered
        if (sv_type == 'INS') and seq_resolved:
            sv_list[len(sv_list)-1].ins_seq = rec.alts[0]
            sv_list[len(sv_list)-1].if_seq_resolved = True
        
    f.close()
    
    #index sv: second_filter: centromere, non-cov
    #third_filter: size
#     for sv in sv_list:
#         second_filter(sv)
#         third_filter(sv)

if __name__ == "__main__":
    main()