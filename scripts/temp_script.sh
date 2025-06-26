
singularity exec --nv --overlay /scratch/lg154/python10/overlay-25GB-500K.ext3:ro \
/scratch/lg154/python10/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash

singularity exec --nv --overlay /scratch/lg154/python10/overlay-25GB-500K.ext3:rw \
/scratch/lg154/python10/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash

conda activate py310

#### analysis for the centroid of 48kHZ model, tensor_cut=48000, channel=2
python analysis.py \
    distributed.data_parallel=False \
    common.save_interval=1 \
    common.test_interval=1 \
    common.max_epoch=100 \
    common.log_interval=1000 \
    datasets.tensor_cut=48_000 \
    datasets.batch_size=16 \
    datasets.num_workers=18 \
    lr_scheduler.warmup_epoch=1 \
    model.sample_rate=48_000 \
    model.causal=False \
    model.norm=time_group_norm \
    model.segment=1. \
    model.name=encodec_48khz_reproduce \
    model.channels=2 \
    model.train_discriminator=0.5 \
    balancer.weights.l_g=4 \
    balancer.weights.l_feat=4 \
    optimization.lr=1e-4 \
    optimization.disc_lr=1e-4 \
    checkpoint.resume=True \
    checkpoint.checkpoint_path='/scratch/lg154/sseg/encodec/outputs/2025-06-16/23-30-40/checkpoints/bs16_cut48000_length0_epoch90_lr0.0001.pt' \
    checkpoint.disc_checkpoint_path='/scratch/lg154/sseg/encodec/outputs/2025-06-16/23-30-40/checkpoints/bs16_cut48000_length0_epoch90_disc_lr0.0001.pt'
    