# ChineseMedLLaMA
ChineseMedLLaMA is a Chinese medical assistant developed based on LLaMA-3.

## Requirements
+ python 3.10
+ pytorch 2.1
+ [transformers](https://github.com/huggingface/transformers)
+ [peft](https://github.com/huggingface/peft)
+ [datasets](https://github.com/huggingface/datasets)
+ [deepspeed](https://github.com/microsoft/DeepSpeed)
+ L20 GPU

We recommend to use conda to manage virtual environments:
```
conda env update --name <env> --file requirements.yml
```

## Continual Pre-training LLaMA-3

First preprocess collected medical data as following, 
``` 
cd src/pretrain
bash preprocess.sh
```

then do pretraining:
``` 
cd src/pretrain
pretrain_medllama_bsz512.sh
```

## Instruct Tuning based on LLaMA-3

First preprocess collected medical instruction data as following, 
``` 
cd src/finetune
bash preprocess-clm.sh
```

then do pretraining:
``` 
cd src/finetune
sft_medllama_bsz128.sh          # Full parameter finetuning
peft_medllama_bsz32.sh          # Parameter-efficient finetuning
```