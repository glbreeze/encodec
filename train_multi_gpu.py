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
from utils import (count_parameters, save_master_checkpoint, set_seed,
                   start_dist_train)
from balancer import Balancer

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define train one step function
class Trainer: 
    def __init__(self, local_rank, world_size, config, tmp_file=None):
        self.local_rank = local_rank
        self.world_size = world_size
        self.config = config
        self.tmp_file = tmp_file
        self.resume_epoch = 0

        self._setup_logging()
        self._set_seed()
        self._init_data()

        self._init_models()
        self._init_optimizers_and_schedulers()
        if self.config.checkpoint.resume:
            self._load_checkpoints()
        self._wrap_distributed()
        self.balancer = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
        if self.balancer is not None:
            logger.info(f'Loss balancer with weights {self.balancer.weights} instantiated')

    def _set_seed(self):
        if self.config.common.seed is not None:
            set_seed(self.config.common.seed)

    def _init_data(self):
        self.trainset = data.CustomAudioDataset(config=self.config)
        self.testset = data.CustomAudioDataset(config=self.config, mode='test')
        self.train_sampler = None
        self.test_sampler = None

        if self.config.distributed.data_parallel:
            self._init_distributed()

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.config.datasets.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            collate_fn=collate_fn,
            pin_memory=self.config.datasets.pin_memory)
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.config.datasets.batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=self.config.datasets.pin_memory)
    
    def _init_distributed(self):
        if self.config.distributed.init_method == "tmp":
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="file://{}".format(self.tmp_file),
                rank=self.local_rank,
                world_size=self.world_size)
        elif self.config.distributed.init_method == "tcp":
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '6008')
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=self.local_rank,
                world_size=self.world_size)
        torch.cuda.set_device(self.local_rank)
        torch.cuda.empty_cache()
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.testset)
    
    def _init_models(self):
        self.model = EncodecModel._get_model(
            self.config.model.target_bandwidths,
            self.config.model.sample_rate,
            self.config.model.channels,
            causal=self.config.model.causal,
            model_norm=self.config.model.norm,
            audio_normalize=self.config.model.audio_normalize,
            segment=self.config.model.segment,
            name=self.config.model.name,
            ratios=self.config.model.ratios,
            stagewise=self.config.quantizer.stagewise,
            stage=self.config.quantizer.stage)

        self.disc_model = MultiScaleSTFTDiscriminator(
            in_channels=self.config.model.channels,
            out_channels=self.config.model.channels,
            filters=self.config.model.filters,
            hop_lengths=self.config.model.disc_hop_lengths,
            win_lengths=self.config.model.disc_win_lengths,
            n_ffts=self.config.model.disc_n_ffts)
        
        # log model, disc model parameters and train mode
        logger.info(self.model)
        logger.info(self.disc_model)
        logger.info(self.config)
        logger.info(f"Encodec Model Parameters: {count_parameters(self.model)} | Disc Model Parameters: {count_parameters(self.disc_model)}")
        logger.info(f"model train mode :{self.model.training} | quantizer train mode :{self.model.quantizer.training} ")

        self.model.cuda()
        self.disc_model.cuda()
    
    def _init_optimizers_and_schedulers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        disc_params = [p for p in self.disc_model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam([{'params': params, 'lr': self.config.optimization.lr}], betas=(0.5, 0.9))
        self.optimizer_disc = optim.Adam([{'params': disc_params, 'lr': self.config.optimization.disc_lr}], betas=(0.5, 0.9))

        steps_per_epoch = len(self.trainloader)
        self.scheduler = WarmupCosineLrScheduler(self.optimizer, max_iter=self.config.common.max_epoch * steps_per_epoch,
                                                  eta_ratio=0.1, warmup_iter=self.config.lr_scheduler.warmup_epoch * steps_per_epoch, warmup_ratio=1e-4)
        self.disc_scheduler = WarmupCosineLrScheduler(self.optimizer_disc, max_iter=self.config.common.max_epoch * steps_per_epoch,
                                                      eta_ratio=0.1, warmup_iter=self.config.lr_scheduler.warmup_epoch * steps_per_epoch, warmup_ratio=1e-4)

        self.scaler = GradScaler() if self.config.common.amp else None
        self.scaler_disc = GradScaler() if self.config.common.amp else None
    
    def _load_checkpoints(self):
        assert self.config.checkpoint.checkpoint_path, "Model checkpoint path is empty"
        assert self.config.checkpoint.disc_checkpoint_path, "Discriminator checkpoint path is empty"
        model_ckpt = torch.load(self.config.checkpoint.checkpoint_path, map_location='cpu')
        disc_ckpt = torch.load(self.config.checkpoint.disc_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model_state_dict'])
        self.disc_model.load_state_dict(disc_ckpt['model_state_dict'])
        self.resume_epoch = model_ckpt['epoch']
        if self.resume_epoch >= self.config.common.max_epoch:
            raise ValueError(f"resume epoch {self.resume_epoch} exceeds max_epoch {self.config.common.max_epoch}")
        logger.info(f"Resumed from epoch {self.resume_epoch}")

        if 'scheduler_state_dict' in model_ckpt and 'scheduler_state_dict' in disc_ckpt:
            self.optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(model_ckpt['scheduler_state_dict'])
            self.optimizer_disc.load_state_dict(disc_ckpt['optimizer_state_dict'])
            self.disc_scheduler.load_state_dict(disc_ckpt['scheduler_state_dict'])
            logger.info(f"Resumed optimizer and scheduler states from epoch {self.resume_epoch}")

    def _wrap_distributed(self):
        if self.config.distributed.data_parallel:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.disc_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.disc_model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                broadcast_buffers=False, find_unused_parameters=self.config.distributed.find_unused_parameters)
            self.disc_model = torch.nn.parallel.DistributedDataParallel(
                self.disc_model, device_ids=[self.local_rank], output_device=self.local_rank,
                broadcast_buffers=False, find_unused_parameters=self.config.distributed.find_unused_parameters)
        
    def train(self):

        self.test(0)
        for epoch in range(max(1, self.resume_epoch + 1), self.config.common.max_epoch + 1):
            self.train_one_step(epoch)

            if epoch % self.config.common.test_interval == 0:
                self.test(epoch)

            if epoch % self.config.common.save_interval == 0:
                model_to_save = self.model.module if self.config.distributed.data_parallel else self.model
                disc_model_to_save = self.disc_model.module if self.config.distributed.data_parallel else self.disc_model
                if not self.config.distributed.data_parallel or dist.get_rank() == 0:
                    save_master_checkpoint(epoch, model_to_save, self.optimizer, self.scheduler,
                                           f'{self.config.checkpoint.save_location}epoch{epoch}_lr{self.config.optimization.lr}.pt')
                    save_master_checkpoint(epoch, disc_model_to_save, self.optimizer_disc, self.disc_scheduler,
                                           f'{self.config.checkpoint.save_location}epoch{epoch}_disc_lr{self.config.optimization.lr}.pt')

        if self.config.distributed.data_parallel:
            dist.destroy_process_group()
    
    def train_one_step(self, epoch):
        """train one step function

        Args:
            epoch (int): current epoch
            optimizer (_type_) : generator optimizer
            optimizer_disc (_type_): discriminator optimizer
            model (_type_): generator model
            disc_model (_type_): discriminator model
            trainloader (_type_): train dataloader
            config (_type_): hydra config file
            scheduler (_type_): adjust generate model learning rate
            disc_scheduler (_type_): adjust discriminator model learning rate
            warmup_scheduler (_type_): warmup learning rate
        """
        self.model.train()
        self.disc_model.train()
        data_length=len(self.trainloader)
        # Initialize variables to accumulate losses  
        accumulated_loss_g = 0.0
        accumulated_losses_g = defaultdict(float)
        accumulated_loss_w = 0.0
        accumulated_loss_disc = 0.0

        for idx,input_wav in enumerate(self.trainloader):
            # warmup learning rate, warmup_epoch is defined in config file,default is 5
            input_wav = input_wav.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
            self.optimizer.zero_grad()
            with autocast(enabled=self.config.common.amp):
                output, loss_w, _ = self.model(input_wav) #output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1] 
                logits_real, fmap_real = self.disc_model(input_wav)
                logits_fake, fmap_fake = self.disc_model(output)
                losses_g = total_loss(
                    fmap_real, 
                    logits_fake, 
                    fmap_fake, 
                    input_wav, 
                    output, 
                    sample_rate=self.config.model.sample_rate,
                ) 
            if self.config.common.amp: 
                loss = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f']  + loss_w
                # not implementing loss balancer in this section, since they say amp is not working anyway:
                # https://github.com/ZhikangNiu/encodec-pytorch/issues/21#issuecomment-2122593367
                self.scaler.scale(loss).backward()  
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
                self.scaler.step(self.optimizer)  
                self.scaler.update()   
                # BUG: doesn't this get done later anyway?
                self.scheduler.step()  
            else:
                # They say they use multiple backwards calls, and lambda_w is 1...
                # https://github.com/facebookresearch/encodec/issues/20
                if self.balancer is not None:
                    self.balancer.backward(losses_g, output, retain_graph=True)
                    # naive loss summation for metrics below
                    loss_g = sum([l * self.balancer.weights[k] for k, l in losses_g.items()])
                else:
                    # without balancer: loss = 3*l_g + 3*l_feat + (l_t / 10) + l_f
                    # loss_g = torch.tensor([0.0], device='cuda', requires_grad=True)
                    loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f'] 
                    loss_g.backward()
                loss_w.backward()
                self.optimizer.step()

            # Accumulate losses  
            accumulated_loss_g += loss_g.item()
            for k, l in losses_g.items():
                accumulated_losses_g[k] += l.item()
            accumulated_loss_w += loss_w.item()

            # only update discriminator with probability from paper (configure)
            self.optimizer_disc.zero_grad()
            train_discriminator = torch.BoolTensor([self.config.model.train_discriminator 
                                and epoch >= self.config.lr_scheduler.warmup_epoch 
                                and random.random() < float(self.config.model.train_discriminator)]).cuda()
            # fix https://github.com/ZhikangNiu/encodec-pytorch/issues/30
            if dist.is_initialized():
                dist.broadcast(train_discriminator, 0)

            if train_discriminator:
                with autocast(enabled=self.config.common.amp):
                    logits_real, _ = self.disc_model(input_wav)
                    logits_fake, _ = self.disc_model(output.detach()) # detach to avoid backpropagation to model
                    loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
                if self.config.common.amp: 
                    self.scaler_disc.scale(loss_disc).backward()
                    # torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)    
                    self.scaler_disc.step(self.optimizer_disc)  
                    self.scaler_disc.update()  
                else:
                    loss_disc.backward() 
                    self.optimizer_disc.step()

                # Accumulate discriminator loss  
                accumulated_loss_disc += loss_disc.item()
            self.scheduler.step()
            self.disc_scheduler.step()

            if (not self.config.distributed.data_parallel or dist.get_rank() == 0) and (idx % self.config.common.log_interval == 0 or idx == data_length - 1): 

                step = (epoch - 1) * data_length + idx
                log_data = {
                    "Train/Loss_G": accumulated_loss_g / (idx + 1),
                    "Train/Loss_W": accumulated_loss_w / (idx + 1),
                    "Train/LR_G": self.optimizer.param_groups[0]['lr'],
                    "Train/LR_D": self.optimizer_disc.param_groups[0]['lr'],
                }
                for k, l in accumulated_losses_g.items():
                    log_data[f'Train/{k}'] = l / (idx + 1)

                if self.config.model.train_discriminator and epoch >= self.config.lr_scheduler.warmup_epoch:
                    log_data["Train/Loss_Disc"] = accumulated_loss_disc / (idx + 1)

                wandb.log(log_data, step=step)

                log_msg = (
                    f"Epoch {epoch} {idx+1}/{data_length}\t"
                    f"Avg loss_G: {log_data['Train/Loss_G']:.4f}\t"
                    f"Avg loss_W: {log_data['Train/Loss_W']:.4f}\t"
                    f"lr_G: {log_data['Train/LR_G']:.6e}\t"
                    f"lr_D: {log_data['Train/LR_D']:.6e}"
                )
                if "Train/Loss_Disc" in log_data:
                    log_msg += f"\tloss_disc: {log_data['Train/Loss_Disc']:.4f}"
                logger.info(log_msg)

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        for idx, input_wav in enumerate(self.testloader):
            input_wav = input_wav.cuda()

            output = self.model(input_wav)
            logits_real, fmap_real = self.disc_model(input_wav)
            logits_fake, fmap_fake = self.disc_model(output)
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
            losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output) 

        if not self.config.distributed.data_parallel or dist.get_rank()==0:
            total_loss_g = sum([l.item() for l in losses_g.values()])

            log_data = {
                "Test/Loss_G_Total": total_loss_g,
                "Test/Loss_Disc": loss_disc.item(),
            }
            for k, l in losses_g.items():
                log_data[f"Test/{k}"] = l.item()
            wandb.log(log_data, step=epoch * len(self.trainloader))

            logger.info(f"| TEST | Epoch {epoch} | " + ', '.join(f"{k}={v:.4f}" for k, v in log_data.items()))

            # save a sample reconstruction (not cropped)
            input_wav, _ = self.testloader.dataset.get()
            input_wav = input_wav.cuda()
            output = self.model(input_wav.unsqueeze(0)).squeeze(0)
            # summarywriter can't log stereo files 😅 so just save examples
            sp = Path(self.config.checkpoint.save_folder)
            torchaudio.save(sp/f'GT.wav', input_wav.cpu(), self.config.model.sample_rate)
            torchaudio.save(sp/f'Reconstruction.wav', output.cpu(), self.config.model.sample_rate)
    
    def _setup_logging(self):
        # remove the logging handler "somebody" added
        logger.handlers.clear()
        # set logger
        file_handler = logging.FileHandler(f"{self.config.checkpoint.save_folder}/train_encodec_bs{self.config.datasets.batch_size}_lr{self.config.optimization.lr}.log")
        formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
        file_handler.setFormatter(formatter)
        # print to screen
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        # add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    @torch.no_grad()
    def analyze_codebook_stats(self):
        from collections import defaultdict

        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()

        quantizer = model.quantizer
        num_layers = model.quantizer.n_q
        dim = quantizer.dimension
        device = next(model.parameters()).device

        codebook_embeddings = [layer.codebook.data.cpu() for layer in quantizer.vq.layers]

        # Collect stats
        sums = defaultdict(lambda: torch.zeros(dim, device=device))
        sums_sq = defaultdict(lambda: torch.zeros(dim, device=device))
        counts = defaultdict(int)
        index_map = defaultdict(list)
        embeddings = []
        global_index = 0

        for batch in self.trainloader:
            x = batch.cuda()
            encoded_frames = model.encode(x)  # [B, T, D]
            quantized_frames = model.quantize(encoded_frames)  # returns [(codes, scale), ...]

            for (codes, _), (emb, _) in zip(quantized_frames, encoded_frames):
                # emb: [B, D, T], emb_flat: [B*T, D]
                B, K, T = codes.shape
                emb_flat = emb.permute(0, 2, 1).reshape(-1, dim)  # [B*T, D]
                composite_codes = codes.permute(0, 2, 1).reshape(-1, K)  # [B*T, K]
                embeddings.append(emb_flat.cpu())

                for i, (code_tuple, vec) in enumerate(zip(composite_codes.tolist(), emb_flat)):
                    # Generate all prefixes: (c₀,), (c₀,c₁), ..., (c₀,...,cₖ₋₁)
                    for j in range(1, 3): # len(code_tuple)+1
                        key = tuple(code_tuple[:j])
                        sums[key] += vec
                        sums_sq[key] += vec ** 2
                        counts[key] += 1
                        index_map[key].append(global_index + i )
                global_index += emb_flat.shape[0]
                if global_index >= 90000: 
                    break

        embeddings_tensor = torch.cat(embeddings, dim=0)  # [N, D]
        
        # Compute stats
        print("\n===== Composite Codebook Statistics =====")
        for code_tuple in sorted(counts.keys()):
            count = counts[code_tuple]
            mean = sums[code_tuple] / count
            var = (sums_sq[code_tuple] / count) - mean.pow(2)
            if len(code_tuple)==1:
                print(f"Code {code_tuple}: count={count:6d}, mean_norm={mean.norm():.4f}, avg_var={var.mean().item():.4f}")
        
        print("====== save the result ======")
        serializable_index_map = {str(k): v for k, v in index_map.items()}
        torch.save({
            "embeddings": embeddings_tensor,
            "index_map": serializable_index_map,
            "codebook_embeddings": codebook_embeddings
        }, "codebook_stats1.pth")
        logger.info(f"save file to codebook_stats1.pth")



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

    wandb.login()
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.init(project='encodec', name=config.common.exp_name)
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    wandb.config.update(cfg_dict)


    # set distributed
    def run_trainer(local_rank, world_size, config, tmp_file=None):
        trainer = Trainer(local_rank, world_size, config, tmp_file)
        trainer.train()

    if config.distributed.data_parallel:  
        world_size = config.distributed.world_size  
        if config.distributed.init_method == "tmp":  
            import tempfile  
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:  
                start_dist_train(run_trainer, world_size, config, tmp_file.name)  
        elif config.distributed.init_method == "tcp":  
            start_dist_train(run_trainer, world_size, config)  
    else:  
        run_trainer(0, 1, config)  # set single gpu train 


if __name__ == '__main__':
    main()
