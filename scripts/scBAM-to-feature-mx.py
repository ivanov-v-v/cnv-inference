#==================================================================================#
# This script takes scBAM file with a list of cell barcodes as an input,           #
# carries out the genome binning and produces the matrix of gene counts.           #
# This matrix comes in .tsv format with the following columns names:               #
# chromosome    bin_start   bin_end cell_barcode_1  cell_barcode_2  ...            #
#==================================================================================#
# Currently each barcode is processed separately. As there may be thousands        #
# of unique barcodes in each sample, this makes the BAM splitting a bottleneck.    #
# We are thinking about processing barcodes in groups using prior information      #
# (for example, clusters found by Louvain's algorithm and outputted by CellRanger).#
# In order to run this script, please, download the subset-bam binary from github  #
# repository of 10XGenomics: https://github.com/10XGenomics/subset-bam — and place #
# the binary in the same directory! If it's not there, it will be downloaded.      #
#==================================================================================#
# In order to achieve peak performance, follow the guidelines: if you have N cores,#
# then batch_size must be set to be about N/16 or N/8. As BAM splitting is a hard  #
# operation, it is important to run as many simultaneous splits in parallel        #
# as possible. Processing of each barcode is distributed evenly over the cores.    #
#==================================================================================#
# Current version of the script processes 4 barcodes from 30gb file with 5k cells  #
# in 10 minutes. Performance improvements are possible.                            #
#==================================================================================#

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

N_CORES = mp.cpu_count() # number of available cores

parser = argparse.ArgumentParser(description='Provide the .bam file and the list of unique cell barcodes '
                                             'outputted by CellRanger together with the desired bin width. '
                                             'Requires subset-bam from 10XGenomics, in binary: https://github.com/10XGenomics/subset-bam. '
                                             'Download the file and put it in the same drirectory.')
parser.add_argument('bam', type=str, help=".bam from CellRanger\'s output")
parser.add_argument('cb', type=str, help="list of unique cell barcodes from CellRanger\'s output")
parser.add_argument('-o', '--out', type=str, default=None, help="location of the outputted matrix")
parser.add_argument('-w', '--width', type=int, default=1000000, help='width of a bin (in bp)')
parser.add_argument('-v', '--verbose',  help="display progress bars", action="store_true")
parser.add_argument('-s', '--batch_size', type=int, default=4, help='barcodes will be processed in chunks of this size')

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
if not os.path.exists("subset-bam"):
    os.system("wget 'https://github.com/10XGenomics/subset-bam/releases/download/1.0/subset-bam-1.0-x86_64-linux.tar.gz'")
    os.system("gunzip subset-bam-1.0-x86_64-linux.tar.gz")
    os.system("mv subset-bam-1.0-x86_64-linux/subset-bam .")
    os.system("rm -rf subset-bam-1.0-x86_64-linux")


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

        gen = range(0, n_cb, args.batch_size)
        if args.verbose:
            gen = tqdm(gen, desc="batch of cell barcodes")
        for i in gen:
            cb_batch = cb_lst[i : i + args.batch_size]
            pool = mp.Pool(args.batch_size)
            pool.map(feature_mx_from_barcode, cb_batch)
            pool.close()
            pool.join()

        # and then we merge those vectors into feature matrix
        result_df = pd.read_csv("{}/cb_{}.bedg".format(tmp_dir, cb_lst[0]), sep='\t',
                                names=["chromosome", "bin_start", "bin_end", cb_lst[0]])
        result_tsv = "read_fmx_{}.tsv".format(sample_name) if args.out is None else args.out

        gen = cb_lst[1:]
        if args.verbose:
            gen = tqdm(gen, desc=".bedg to merge")
        for cb in gen:
            tmp_df = pd.read_csv('{}/cb_{}.bedg'.format(tmp_dir, cb), sep='\t',
                                names=["chromosome", "bin_start", "bin_end", cb])        
            result_df[cb] = tmp_df[cb]
            result_df.to_csv(result_tsv, index=False, sep='\t')

finally:
    os.system("rm -rf {}".format(tmp_dir))


