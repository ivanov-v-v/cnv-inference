#!/usr/bin/bash

# This script takes .bam file as an input and extracts all unique cell barcodes present in that file.
# It processes each chromosome separately (in parallel) and then merges the results. 
# It also prints the chromosome name when done. Redirect the output to /dev/null to suppress it.

declare -i MAX_FORKS=8
INFILE=$1
FNAME=$(echo $1 | sed 's/.*\///')
echo "processing $INFILE"

tmp_dir=$RANDOM\-chr\-dir
mkdir $tmp_dir

function chrom_from_bam() {
       samtools idxstats $1 | awk '{print $1}' | grep -v "\*"
}

function cb_on_chrom() {
       samtools view $1 $2 | grep -Po "CB:Z:\K[A-Z]*(\-\d+){0,1}" | sort -uo "$tmp_dir/$2.txt"
}

chromosome_list=$(chrom_from_bam $1)
declare -i forks_running=0
for chrom in $chromosome_list; do
	echo $chrom 
	forks_running+=1
	if [ $forks_running -eq $MAX_FORKS ]; then 
		wait
		forks_running=0
	fi
	cb_on_chrom $INFILE $chrom & # here the parallelization happens
done
wait

cat $tmp_dir/*.txt | sort --parallel 8 -uo barcodes_$FNAME.txt
rm -rf $tmp_dir	

