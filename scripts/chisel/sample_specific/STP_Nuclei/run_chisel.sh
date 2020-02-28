#!/usr/bin/bash

GROUP_DIR=/icgc/dkfzlsdf/analysis/B260                                                                                                                                      
PROJECT_DIR=$GROUP_DIR/projects/chromothripsis_medulloblastoma
RESOURCE_DIR=$GROUP_DIR/resource    
CHISEL_DIR=$SLAVA/chisel
OUTPUT_DIR=$CHISEL_DIR/outs/STP-Nuclei

NORMAL_BARCODES=${CHISEL_DIR}/data/STP-Nuclei/normal_cell_barcodes.txt
TUMOROUS_BARCODES=${CHISEL_DIR}/data/STP-Nuclei/tumorous_cell_barcodes.txt

rm -rf $OUTPUT_DIR/*
mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR

#TUMOUR_BAM=/icgc/dkfzlsdf/analysis/B260/projects/chromothripsis_medulloblastoma/data/10xCNV/STP/STP-PDX/outs/possorted_bam.bam
#NORMAL_BAM=/icgc/dkfzlsdf/analysis/B260/projects/chromothripsis_medulloblastoma/data/10xCNV/STP/STP-Nuclei/outs/possorted_bam.bam

TUMOUR_BAM=$CHISEL_DIR/data/STP-Nuclei/tumour_possorted_bam.bam 
NORMAL_BAM=$CHISEL_DIR/data/STP-Nuclei/normal_possorted_bam.bam
PHASING_VCF=$PROJECT_DIR/eagle2_phasing/rawdata/EAGLE2_20190909_vcfs/all.vcf
REF_GENOME_FA=$CHISEL_DIR/data/STP-Nuclei/genome.fa

du -bh $NORMAL_BAM
du -bh $TUMOUR_BAM

#samtools idxstats $TUMOUR_BAM | awk '{print $1}' | grep -v "\*" > $OUTPUT_DIR/chromosomes.txt
#CHROMOSOMES="$(cat $OUTPUT_DIR/chromosomes.txt | sed 's/\n/ /g')"
#echo $CHROMOSOMES
CHROMOSOMES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"

python2.7 $CHISEL_DIR/bin/chisel.py \
    --tumor $TUMOUR_BAM \
    --normal $NORMAL_BAM \
    --reference $REF_GENOME_FA \
    --listphased $PHASING_VCF \
    --chromosomes "${CHROMOSOMES}" \
    --rundir $OUTPUT_DIR \
    --jobs 16 \
    --seed 25
