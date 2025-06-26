#!/bin/bash

#SBATCH --job-name=encodec_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,h100_1,rtx8000

# Singularity path
ext3_path=/scratch/$USER/python11/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/python11/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv --overlay /scratch/lg154/python11/overlay-25GB-500K.ext3:ro \
/scratch/lg154/python11/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh
export SSL_CERT_FILE=/scratch/lg154/sseg/fs-ood/cacert.pem

python analysis.py \
        --config-name config_tree \
        distributed.data_parallel=False \
        common.save_interval=1 \
        common.test_interval=1 \
        common.max_epoch=100 \
        common.log_interval=1000 \
        datasets.tensor_cut=142_800 \
        datasets.batch_size=16 \
        datasets.num_workers=18 \
        lr_scheduler.warmup_epoch=1 \
        model.sample_rate=24_000 \
        model.causal=False \
        model.norm=time_group_norm \
        model.segment=1. \
        model.name=encodec_24khz_reproduce \
        model.channels=1 \
        model.train_discriminator=0.5 \
        balancer.weights.l_g=4 \
        balancer.weights.l_feat=4 \
        optimization.lr=1e-4 \
        optimization.disc_lr=1e-4 \
        checkpoint.resume=True \
        checkpoint.checkpoint_path='/scratch/lg154/sseg/encodec/outputs/2025-06-23/tree_rvq/checkpoints/bs16_cut47760_length0_epoch45_lr0.0001.pt' \
        checkpoint.disc_checkpoint_path='/scratch/lg154/sseg/encodec/outputs/2025-06-23/tree_rvq/checkpoints/bs16_cut47760_length0_epoch45_disc_lr0.0001.pt'
        
"

# model.target_bandwidths="[3., 6., 12., 24.]" \