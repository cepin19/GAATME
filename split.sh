#!/bin/bash

# Split source.txt and reference.txt into 15-line chunks
split -d -l 15 --suffix-length=3   <(head -n 2455 sen-wmt24_encs.en) chunks/source_chunk_
#split -d -l 15  --suffix-length=3  ../ chunks/reference_chunk_

# Calculate the number of chunks
num_chunks=$(ls -l chunks/source_chunk_* | wc -l)

# Split hyps.txt into corresponding  line chunks
split --suffix-length=3  -d -l $((15 * 39)) sen-wmt24_encs-sent.txt  chunks/hyps_chunk_

# Rename hyps_chunk files to match source_chunk and reference_chunk names
#for ((i=0; i<num_chunks; i++)); do
#  mv "hyps_chunk_$(printf "%03d" $i)" "hyps_chunk_$((i+1))"
#done
