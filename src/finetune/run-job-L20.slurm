#!/bin/bash
#SBATCH --job-name=xfbai-benchmark     #作业名称
#SBATCH --partition=q_intel_share_L20001      #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=32            #申请每个节点32核CPU
#SBATCH --gres=gpu:2                    #申请4张GPU卡
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
module load anaconda3/2023.09-0 cuda/11.8.0 gcc/9.3.0
source activate py38torch2.0vllm
cd $SLURM_SUBMIT_DIR
bash peft_medllama_bsz32.sh
