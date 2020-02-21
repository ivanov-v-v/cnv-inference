#!/usr/bin/bash

GROUP_DIR=/icgc/dkfzlsdf/analysis/B260
PROJECT_DIR=${GROUP_DIR}/projects/chromothripsis_medulloblastoma
INPUT_BAM=${PROJECT_DIR}/data/10xCNV/STP/STP-Nuclei/outs/possorted_bam.bam
CHISEL_HOME=${SLAVA}/chisel

NORMAL_BARCODES=${CHISEL_HOME}/data/STP-Nuclei/normal_cell_barcodes.txt
TUMOROUS_BARCODES=${CHISEL_HOME}/data/STP-Nuclei/tumorous_cell_barcodes.txt

NORMAL_OUTPUT_BAM=${CHISEL_HOME}/data/STP-Nuclei/normal_possorted_bam.bam
TUMOROUS_OUTPUT_BAM=${CHISEL_HOME}/data/tumorous_possorted_bam.bam

./subset_bam.sh ${INPUT_BAM} ${NORMAL_BARCODES} ${NORMAL_OUTPUT_BAM}
./subset_bam.sh ${INPUT_BAM} ${TUMOROUS_BARCODES} ${TUMOROUS_OUTPUT_BAM}
