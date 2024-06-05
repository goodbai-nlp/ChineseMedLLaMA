#!/bin/bash
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $ROOT_DIR
BasePath=/UNICOMFS/hitsz_khchen_1

MODEL=$1
DATA=${BasePath}/data-bak/TaskData/pubmedqa-sim/pqal_test.jsonl

python -u inference_vllm.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 32 \
    --out_prefix "test-pred5" \
    --input_key "input" \
    --sys_instruction "As a doctor, please address the medical inquiries based on the patient's account. Ensure your responses are concise and straightforward." \
    --instruction "Analyze the question given its context and answer with yes, no or maybe directly." \
    --prompt_template "chat-v1" 2>&1 | tee $MODEL/eval.log
