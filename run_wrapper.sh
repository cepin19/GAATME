#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
. /home/jon/miniconda3//etc/profile.d/conda.sh
conda activate /lnet/work/people/jon/comet2
export TOKENIZERS_PARALLELISM=false
export HOME=/lnet/work/people/jon/comet2
python3 ../comet_mbr.py $@
