#!/bin/bash
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $ROOT_DIR
BasePath=/UNICOMFS/hitsz_khchen_1

MODEL=$1
DATA=${BasePath}/data/TaskData/medmcqa-processed/dev.jsonl

python -u inference_hf.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 32 \
    --out_prefix "test-pred" \
    --input_key "input" \
    --sys_instruction "You're a doctor, kindly address the medical queries according to the patient's account." \
    --instruction "Please respond with A, B, C or D. The answer to the question is:" \
    --prompt_template "chat-v1" 2>&1 | tee $MODEL/eval.log

python -u medmcqa_eval.py ${DATA} $MODEL/usmle_valid_pred_vllm 2>&1 | tee $MODEL/eval-usmle-valid.log