import os
import random
from pathlib import Path

import librosa
import torchaudio
import pandas as pd
import torch
import audioread
import torchaudio.functional as F

import logging
logger = logging.getLogger(__name__)

from utils import convert_audio

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, root_path ='/scratch/lg154/sseg/encodec', transform=None, mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        if mode == 'train':
            self.audio_files = pd.read_csv(config.datasets.train_csv_path,on_bad_lines='skip')
        elif mode == 'test':
            self.audio_files = pd.read_csv(config.datasets.test_csv_path,on_bad_lines='skip',)
        self.transform = transform
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels
        self.root_path = root_path

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)  

    def get(self, idx=None):
        """uncropped, untransformed getter with random sample feature"""
        if idx is not None and idx > len(self.audio_files):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        
        rel_path = self.audio_files.iloc[idx, 0]
        abs_path = (Path(self.root_path) / rel_path).resolve()

        try:
            logger.debug(f'Loading {abs_path}')
            # waveform, sample_rate = librosa.load(str(abs_path), sr=self.sample_rate, mono=self.channels == 1)
            waveform, sr = torchaudio.load(str(abs_path)) # shape: [channels, time]
        except (audioread.exceptions.NoBackendError, ZeroDivisionError):
            logger.warning(f"Not able to load {abs_path}, removing from dataset")
            self.audio_files.drop(idx, inplace=True)
            return self[idx]
        if sr != self.sample_rate:
            waveform = F.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        
        # convert to mono if needed
        if self.channels == 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif self.channels == 2 and waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)

        return waveform, self.sample_rate

    def __getitem__(self, idx):
        # waveform, sample_rate = torchaudio.load(self.audio_files.iloc[idx, :].values[0])
        # """you can preprocess the waveform's sample rate to save time and memory"""
        # if sample_rate != self.sample_rate:
        #     waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        waveform, sample_rate = self.get(idx)

        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1) # random start point
                waveform = waveform[:, start:start+self.tensor_cut] # cut tensor
                return waveform, sample_rate
            else:
                return waveform, sample_rate


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors