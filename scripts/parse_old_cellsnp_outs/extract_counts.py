#!$SLAVA/.conda/envs/xclone/bin/python

import sys

def extract_counts(fields):
    for i in range(len(fields)):
        tokenized = fields[i].split(":")
        if len(tokenized) > 1:
            fields[i] = (','.join(tokenized[1:3])).replace('.', '')
    return fields

if __name__ == "__main__":
    for line in sys.stdin:
        fields = line.strip().split('\t')
        if fields[0] == "CHROM": # indicates colnames row
            expected_fields_count = len(fields)
            sys.stdout.write(line)
            continue
        
        if len(fields) != expected_fields_count:
        # omit malformed lines
            continue

        fields = extract_counts(fields)
        output = "{}\n".format('\t'.join(fields))
#        sys.stderr.write(output)
        sys.stdout.write(output)
