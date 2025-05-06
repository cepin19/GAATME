#!/bin/bash


#SBATCH -J ga-beam-samp-nodict-mbr-comet+qe+bleu					  # name of job
#SBATCH -p gpu-troja					  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/ga_test_git/GAATME 
#SBATCH --gres=gpu:1    
#SBATCH  --constraint="gpuram11G|gpuram40G|gpuram24G|gpuram48G"
#SBATCH  --mem=15G
#SBATCH --cpus-per-task=2

lines=0
bleu_w=1
chrf_w=1
qe_w=1
mbr_w=1
fn="wmt24.encs.head10"
num_hypotheses=25
out_dir=out
mkdir -p $out_dir/
# Handling continuation after unexpected end
# Iterate through the files in the directory
for file in $out_dir/$fn.out_head+* ; do
  # Read the line count of the current file and add it to the total line count, this is for continuing if the job ends unexpectedly before processing the whole file
  lines=$((lines + $(wc -l < "$file")))
done
     hyplines=$((lines * num_hypotheses + 1)) 
     h=$((lines+1))
infile="$fn"_tail+"$h".in
tail -n+$h  $fn.src >"$infile"
hypfile="$fn"_tail+"$h".hyps
tail -n+$hyplines $fn.hyps  >"$hypfile"



bash run_wrapper.sh -s "$infile" -t "$hypfile" -p "$hypfile" -f fitness_comet_mbr_and_qe_and_bleu_and_chrf_w_multiref --bleu_w=$bleu_w --chrf_w=$chrf_w  --comet_qe_w=$qe_w --comet_mbr_w=$mbr_w --model-qe Unbabel/wmt22-cometkiwi-da --model Unbabel/wmt22-comet-da --num_samples $num_hypotheses --num_pseudo_refs $num_hypotheses -g 300 -m 1 -c 0.1 -l $out_dir/$fn.log_head+$h >  $out_dir/$fn.out_head+$h 2> $out_dir/$fn.err_head+$h
