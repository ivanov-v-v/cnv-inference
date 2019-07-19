import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

N_CORES = mp.cpu_count() # number of available cores

parser = argparse.ArgumentParser(description='provide the .bam file and the list of unique cell barcodes '
                                             'outputted by CellRanger together with the desired bin width')
parser.add_argument('--bam', type=str, help=".bam from CellRanger\'s output")
parser.add_argument('--cb', type=str, help="list of unique cell barcodes from CellRanger\'s output")
parser.add_argument('--width', type=int, default=1000000, help='width of a bin (in bp)')
#parser.add_argument('--verbose', nargs=0, type=bool, default=True, help="verbose")
parser.add_argument('--batch_size', type=int, default=4, help='barcodes will be processed in chunks of this size')

args = parser.parse_args()

# maybe there is no need in this conversion actually
args.bam = os.path.abspath(args.bam) 
args.cb = os.path.abspath(args.cb) 

# temporary directory that stores bam files and feature vectors
tmp_dir = ".{}_fmx".format(np.random.rand())
os.system("mkdir {}".format(tmp_dir))

sample_name = args.bam.split('/')[-1]
assert sample_name.split('.')[-1] == 'bam'
sample_name = '.'.join(sample_name.split('.')[:-1])

CORES_PER_JOB = np.ceil(N_CORES / args.batch_size).astype(int)


def feature_mx_from_barcode(cb):
    # subset-bam can't take a list of barcodes as input so we create files
    cb_txt = "{}/cb_{}.txt".format(tmp_dir, cb)
    os.system("echo {} > {}".format(cb, cb_txt))
    # we also need to specify the name for the output BAM file
    cb_bam = "{}/cb_{}.bam".format(tmp_dir, cb)
    os.system("./subset-bam -c {} -b {} -o {} --cores {}".format(cb_txt, args.bam, cb_bam, CORES_PER_JOB))  
    os.system("rm {}".format(cb_txt)) # why keep the file when it's not needed anymore
#    print("{}: finished subsetting".format(cb))

    # file needs to be indexed after subsetting
    os.system("samtools index {}".format(cb_bam))
#    print("{}: finished indexing".format(cb))

    # here the binning and read counting happens
    cb_bedg = "{}/cb_{}.bedg".format(tmp_dir, cb)
    os.system("bamCoverage -b {} -o {} -of bedgraph --binSize {} --numberOfProcessors {}"\
             .format(cb_bam, cb_bedg, args.width, CORES_PER_JOB))
    os.system("rm {}".format(cb_bam)) # saving disk memory
#    print("{}: finished coverage".format(cb))


try:
    with open(args.cb, "r") as cb_file:
        # at first we compute the read count vector for each cell
        cb_lst = cb_file.read().splitlines()
        cb_lst.sort()
        n_cb = len(cb_lst)
        assert n_cb > 0, "empty list of cell barcodes"
        for i in tqdm(range(0, n_cb, args.batch_size), desc="batch of cell barcodes"):	
            cb_batch = cb_lst[i : i + args.batch_size]
            pool = mp.Pool(args.batch_size)
            pool.map(feature_mx_from_barcode, cb_batch)
            pool.close()
            pool.join()

        # and then we merge those vectors into feature matrix
        result_df = pd.read_csv("{}/cb_{}.bedg".format(tmp_dir, cb_lst[0]), sep='\t',
                                names=["chrom", "bin_start", "bin_end", cb_lst[0]])

        result_csv = "read_fmx_{}.csv".format(sample_name)
        for cb in tqdm(cb_lst[1:], desc=".bedg to merge"):
            tmp_df = pd.read_csv('{}/cb_{}.bedg'.format(tmp_dir, cb), sep='\t',
                                names=["chrom", "bin_start", "bin_end", cb])        
            result_df[cb] = tmp_df[cb]
            result_df.to_csv(result_csv, index=False, sep='\t')

finally:
    os.system("rm -rf {}".format(tmp_dir))


