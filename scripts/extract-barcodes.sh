# This script takes .bam file as an input and extracts all unique cell barcodes present in that file.
# It processes each chromosome separately (in parallel) and then merges the results. 
# It also prints the chromosome name when done. Redirect the output to /dev/null to suppress it.

tmp_dir=$1\-chr\-dir
mkdir $tmp_dir

function chrom_from_bam() {
       samtools idxstats $1 | awk '{print $1}' | grep -v "\*"
}

function cb_on_chrom() {
       samtools view $1 $2 | grep -Po "CB:Z:\K[A-Z]*(\-\d+){0,1}" | sort -uo "$tmp_dir/$2.txt"
}

chromosome_list=$(chrom_from_bam $1);
for chrom in $chromosome_list; do
       cb_on_chrom $1 $chrom & # here the parallelization happens
       echo $chrom
done
wait

cat $tmp_dir/*.txt | sort --parallel 8 -uo $1_barcodes.txt
rm -rf $tmp_dir	
