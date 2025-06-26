import os
import wandb 
import random
import logging
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict


import hydra
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchaudio

import customAudioDataset as data
from customAudioDataset import collate_fn
from losses import disc_loss, total_loss
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, set_seed, start_dist_train)
from balancer import Balancer
from train_multi_gpu import Trainer

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@hydra.main(config_path='config', config_name='config')
def main(config):
    # set distributed debug, if you encouter some multi gpu bug, please set torch_distributed_debug=True
    if config.distributed.torch_distributed_debug: 
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
    # disable cudnn
    torch.backends.cudnn.enabled = False
    # Ensure save directory exists
    os.makedirs(config.checkpoint.save_folder, exist_ok=True)

    trainer = Trainer(0, 1, config)
    trainer.analyze_codebook_stats(l=1, folder=os.path.dirname(os.path.dirname(config.checkpoint.checkpoint_path)))


if __name__ == '__main__':
    main()


