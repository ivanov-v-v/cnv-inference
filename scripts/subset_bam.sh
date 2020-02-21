#!/usr/bin/bash
INPUT_BAM=$1
BARCODE_LIST=$2
OUTPUT_BAM=$3

# Save the header lines
samtools view -H $INPUT_BAM > header.sam
# Filter alignments using filter.txt. Use LC_ALL=C to set C locale instead of UTF-8
samtools view $INPUT_BAM | LC_ALL=C grep -F -f $BARCODE_LIST\
| samtools view -Sb > $OUTPUT_BAM
# Return the header back to the file
samtools reheader header.sam $OUTPUT_BAM
rm header.sam
