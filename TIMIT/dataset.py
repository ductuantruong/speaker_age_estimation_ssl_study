from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
import random

class TIMITDataset(Dataset):
    def __init__(self,
    wav_folder,
    hparams,
    is_train=True,
    ):
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        g_id = file.split('_')[0]
        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        age =  self.df.loc[id, 'age']
        
        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            
        a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        
        age = (age - a_mean)/a_std
        
        probability = 0.5
        if self.is_train and random.random() <= probability:
            mixup_idx = random.randint(0, len(self.files)-1)
            mixup_file = self.files[mixup_idx]
            mixup_id = mixup_file.split('_')[0][1:]
            mixup_gender = self.gender_dict[self.df.loc[mixup_id, 'Sex']]
            mixup_age =  self.df.loc[mixup_id, 'age']

            mixup_wav, _ = torchaudio.load(os.path.join(self.wav_folder, mixup_file))

            if(mixup_wav.shape[0] != 1):
                mixup_wav = torch.mean(mixup_wav, dim=0) 
            
            if self.narrow_band:
                mixup_wav = self.resampleUp(self.resampleDown(mixup_wav))

            mixup_age = (mixup_age - a_mean)/a_std
            
            if(mixup_wav.shape[1] < wav.shape[1]):
                cnt = (wav.shape[1]+mixup_wav.shape[1]-1)//mixup_wav.shape[1]
                mixup_wav = mixup_wav.repeat(1,cnt)[:,:wav.shape[1]]
            
            if(wav.shape[1] < mixup_wav.shape[1]):
                cnt = (mixup_wav.shape[1]+wav.shape[1]-1)//wav.shape[1]
                wav = wav.repeat(1,cnt)[:,:mixup_wav.shape[1]]
            
            alpha = 1
            lam = np.random.beta(alpha, alpha)
            
            wav = lam*wav + (1-lam)*mixup_wav
            age = lam*age + (1-lam)*mixup_age
            gender = lam*gender + (1-lam)*mixup_gender
            
        return wav, torch.FloatTensor([age]), torch.FloatTensor([gender])
