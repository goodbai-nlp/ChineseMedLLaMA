#!/bin/bash
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $ROOT_DIR
BasePath=/UNICOMFS/hitsz_khchen_1

MODEL=$1
DATA=${BasePath}/data/TaskData/pubmedqa-sim/pqal_test.jsonl

python -u inference_hf.py \
    --test_file ${DATA} \
    --model_name_or_path ${MODEL} \
    --num_beams 1 \
    --max_new_tokens 32 \
    --out_prefix "test-pred4" \
    --input_key "input" \
    --sys_instruction "You're a doctor, kindly address the medical queries according to the patient's account." \
    --instruction "Analyze the question given its context and answer with yes, no or maybe directly." \
    --prompt_template "chat-v1" 2>&1 | tee $MODEL/eval.log

# python eval_bleu.py ${DATA} ${MODEL}/test-pred_test.jsonl_pred_peft
