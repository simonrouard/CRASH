# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from torch.utils.data.distributed import DistributedSampler
from glob import glob
import numpy as np
import os
import random
import torch
import torchaudio
torchaudio.set_audio_backend("sox")


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, params):
        super().__init__()
        self.filenames = []
        self.audio_length = params['audio_length']
        for path in paths:
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        signal = signal[0, :self.audio_length]
        # renormalize the audio
        scaler = max(signal.max(), -signal.min())
        if scaler > 0:
            signal = signal / scaler
        return {
            'audio': signal
        }


def from_path(data_dirs, params):
    dataset = AudioDataset(data_dirs, params)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        collate_fn=None,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True)
