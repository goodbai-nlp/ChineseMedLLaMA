#!/bin/bash
#SBATCH --job-name=MedLLaMA3-SFT     #作业名称
#SBATCH --partition=q_intel_gpu_nvidia_nvlink       #选择资源分区
#SBATCH -N 1                            #申请计算节点数
#SBATCH --ntasks-per-node=64            #申请每个节点32核CPU
#SBATCH --gres=gpu:8                    #申请4张GPU卡
#SBATCH --nodelist=gpu[001]             #指定优先使用节点
#SBATCH --output=logs/run-job%j.out         #作业标准输出
#SBATCH --error=logs/run-job%j.err          #作业标准报错信息
hostname

################################################################
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HAC=mlx5_2
# export NCCL_SOCKET_IFNAME=ib2
# export NCCL_P2P_DISABLE=1

# export MASTER_ADDR=$(scontrol show hostnames $NODELIST | head -n 1)
# export MASTER_PORT=24534
# scontrol show hostname $SLURM_JOB_NODELIST | sed 's/$/ slots=8/' > hostfile

module load amd/Anaconda/2023.3 amd/cuda/11.8.89 amd/gcc_compiler/9.3.0 intel/nccl/2.18.1-1/cuda-11.8 amd/openmpi/3.1.0
source activate py3.10torch2.1devel
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
cd $SLURM_SUBMIT_DIR
bash sft_medllama_bsz128_new.sh
