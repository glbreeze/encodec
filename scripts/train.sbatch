#!/bin/bash

#SBATCH --job-name=encodec_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

srun --container-name=encodec --container-mounts=/hpi/fs00/share/fg-meinel/datasets/audio:/home/audio,/hpi/fs00/home/leonardo.auri/:/home/leonardo.auri/ --container-workdir=/home/leonardo.auri/encodec-pytorch/ \
        --container-writable \
        python train_multi_gpu.py \
        distributed.data_parallel=False \
        common.save_interval=1 \
        common.test_interval=1 \
        common.max_epoch=105 \
        common.log_interval=1000 \
        datasets.tensor_cut=48_000 \
        datasets.batch_size=16 \
        datasets.num_workers=18 \
        lr_scheduler.warmup_epoch=1 \
        model.sample_rate=48_000 \
        model.target_bandwidths="[3., 6., 12., 24.]" \
        model.causal=False \
        model.norm=time_group_norm \
        model.segment=1. \
        model.name=encodec_48khz_reproduce \
        model.channels=2 \
        model.train_discriminator=0.5 \
        balancer.weights.l_g=4 \
        balancer.weights.l_feat=4 \
        optimization.lr=1e-4 \
        optimization.disc_lr=1e-4