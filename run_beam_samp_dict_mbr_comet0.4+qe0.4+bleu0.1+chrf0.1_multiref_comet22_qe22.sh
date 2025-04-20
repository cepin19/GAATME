#!/bin/bash


#SBATCH -J ga-beam-samp-nodict-mbr-comet+qe+bleu					  # name of job
#SBATCH -p gpu-ms					  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/ga_clean_comet-v2/runwmt24_fix 
#SBATCH --gres=gpu:1    
#SBATCH  --constraint="gpuram11G|gpuram40G|gpuram24G"
#SBATCH  --mem=15G
#SBATCH --cpus-per-task=2


d=$(date +%m_%d_%H_%M) 
chunk=x

# Initialize the line count to zero
lines=0
bleu_w=0.1
chrf_w=0.1
qe_w=0.4
mbr_w=0.4
fn=ga-beam-samp-nodict-mbr-comet"$mbr_w"+qe"$qe_w"+bleu"$bleu_w"+chrf"$chrf_w"_multiref_comet22_qe22_$chunk
# Iterate through the files in the directory
for file in $fn.out_head+* ; do
  # Read the line count of the current file and add it to the total line count
  lines=$((lines + $(wc -l < "$file")))
done
     hyplines=$((lines * 39 + 1)) 
     h=$((lines+1))
infile="$fn"_tail+"$h".in
tail -n+$h  chunks/source_chunk_$chunk >"$infile"
hypfile="$fn"_tail+"$h".hyps
tail -n+$hyplines  chunks/hyps_chunk_$chunk  >"$hypfile"
bash run_wrapper.sh -s "$infile" -t "$hypfile" -p "$hypfile" -f fitness_comet_mbr_and_qe_and_bleu_and_chrf_w_multiref --bleu_w=$bleu_w --chrf_w=$chrf_w  --comet_qe_w=$qe_w --comet_mbr_w=$mbr_w --model-qe Unbabel/wmt22-cometkiwi-da --model Unbabel/wmt22-comet-da --num_samples 39 --num_pseudo_refs 39 -g 200 -m 1 -c 0.1 -l $fn.log_head+$h >  $fn.out_head+$h 2> $fn.err_head+$h
