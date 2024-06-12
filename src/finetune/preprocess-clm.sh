
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

#BasePath=/mnt/data/home/usera6k10
BasePath=/mnt/data/home/usera6k10/code/ChineseMedLLaMA

CurDir=$(cd $(dirname $0);cd ..; pwd)

MODEL_NAME=llama3-8b-instruct

MODEL=${BasePath}/data/pretrained-models/${MODEL_NAME}
DataPath=${BasePath}/data
DataSetName=Chinese-MedQA-IT-llama3

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache


lr=2e-5
NUM_EPOCHS=3
BATCH_SIZE_PER_GPU=64
GRADIENT_ACC_STEPS=1

OUTPUT_DIR=${BasePath}/output/exp.MedLLaMA/Preprocess-${DataSetName}-${MODEL_NAME}-ConditionalGenMode

if [ ! -d ${OUTPUT_DIR} ];then
  mkdir -p ${OUTPUT_DIR}
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

echo ${CurDir}
python ${CurDir}/main.py \
    --deepspeed ${CurDir}/ds_configs/stage1_no_offloading.conf \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --conditional_gen True \
    --max_seq_length 2048 \
    --do_preprocess \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --evaluation_strategy "steps" \
    --logging_steps 100 \
    --greater_is_better False \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --ignore_opt_states True \
    --num_train_epochs ${NUM_EPOCHS} \
    --logging_first_step True \
    --gradient_checkpointing \
    --use_peft False \
    --use_flash_attn True \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --tf32 True \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 128 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache-llama3 \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log
