#!/usr/bin/bash

# Main directories
GROUP_DIR=/icgc/dkfzlsdf/analysis/B260    
PROJECT_DIR=$GROUP_DIR/projects/chromothripsis_medulloblastoma                                                                                                              
RESOURCE_DIR=$GROUP_DIR/resource
CHISEL_DIR=$SLAVA/chisel

# Paths to BAMs
FULL_BAM=$PROJECT_DIR/data/10xCNV/STP/STP-Nuclei/outs/possorted_bam.bam
NORMAL_BAM=$CHISEL_DIR/data/STP-Nuclei/normal_possorted_bam.bam
TUMOUR_BAM=$CHISEL_DIR/data/STP-Nuclei/tumour_possorted_bam.bam

# Extracting barcodes from splitted BAMs
$XC_SCRIPTS/extract_barcodes.sh $NORMAL_BAM
$XC_SCRIPTS/extract_barcodes.sh $TUMOUR_BAM

# Merging them together and sorting
cat barcodes_normal_possorted_bam.bam.txt > merged.tmp
cat barcodes_tumour_possorted_bam.bam.txt >> merged.tmp
cat merged.tmp | sort > merged_barcode_list.txt
rm merged.tmp

# Extracting barcodes from the original BAM
# This takes time, thereby is only done in the very end
$XC_SCRIPTS/extract_barcodes.sh $FULL_BAM
diff -u merged_barcode_list "barcodes_${FULL_BAM}.txt"  | sed -nr 's/^+([^+].*)/\1' > barcodes_diff.txt
