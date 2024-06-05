#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PATH="/opt/conda/bin:$PATH"
source ~/.bashrc
# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# echo $ROOT_DIR

BasePath=/baixuefeng
PORT_ID=$(expr $RANDOM + 1000)

MODEL=$1
DATA=$2
python -u inference_vllm.py --test_file ${DATA} --model_name_or_path ${MODEL} --num_beams 1 --max_new_tokens 128 --out_prefix "test-pred" --instruction "" --prompt_template "pubmedqa" 2>&1 | tee $MODEL/eval.log