#!/bin/bash
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $ROOT_DIR
BasePath=/UNICOMFS/hitsz_khchen_1

MODEL=$1
DATA=${BasePath}/data-bak/TaskData/medmcqa-processed/dev.jsonl

python -u inference_vllm.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 32 \
    --out_prefix "test-pred-medmcqa" \
    --input_key "input" \
    --sys_instruction "As a doctor, please address the medical inquiries based on the patient's account. Ensure your responses are concise and straightforward." \
    --instruction "Please respond with A, B, C or D." \
    --prompt_template "chat-v1" 2>&1 | tee $MODEL/eval.log
