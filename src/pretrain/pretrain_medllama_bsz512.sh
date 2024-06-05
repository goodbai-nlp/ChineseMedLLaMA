export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#BasePath=/data1/xfbai
BasePath=/home/export/base/ycsc_chenkh/chenkh_nvlink/online1/xfbai

CurDir=$(cd $(dirname $0);cd ..; pwd)

MODEL_NAME=llama3-8b
#MODEL_NAME=llama2-7b

MODEL=${BasePath}/data/pretrained-models/${MODEL_NAME}
DataPath=${BasePath}/data
DataSetName=pubmed-abs

export HF_DATASETS_CACHE=${DataPath}/${DataSetName}/.cache

MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=64
TOTAL_BATCH_SIZE=512
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

lr=2e-5
NUM_EPOCHS=3

OUTPUT_DIR=${BasePath}/output/exp.MedLLaMA3/DAPT-${DataSetName}-${MODEL_NAME}-GenMode-lr-${lr}-totalbsz${TOTAL_BATCH_SIZE}-decay0.1-${NUM_EPOCHS}epoch

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
deepspeed ${CurDir}/main.py \
    --deepspeed ${CurDir}/ds_configs/stage3_no_offloading.conf \
    --data_path ${DataPath}/${DataSetName} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name ${MODEL} \
    --use_fast_tokenizer False \
    --conditional_gen False \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --evaluation_strategy "no" \
    --logging_steps 200 \
    --greater_is_better False \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 20 \
    --save_only_model True \
    --num_train_epochs ${NUM_EPOCHS} \
    --logging_first_step True \
    --gradient_checkpointing \
    --use_peft False \
    --use_flash_attn True \
    --output_dir ${OUTPUT_DIR} \
    --torch_dtype "bfloat16" \
    --bf16 \
    --tf32 True \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --data_cache_dir ${DataPath}/${DataSetName}/.cache-llama3 \
    --wandb_project "medllama3" \
    --report_to "tensorboard" 2>&1 | tee ${OUTPUT_DIR}/training.log
