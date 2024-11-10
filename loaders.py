#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, hdf5storage
from torch.utils.data import Dataset
import numpy as np

class Channels(Dataset):
    """MIMO Channels"""

    def __init__(self, seed, config, norm=None):
        # Get spacings
        target_spacings = config.data.spacing_list
        target_channel  = config.data.channel
        target_db       = config.db
        # Output channels
        self.spacings  = np.copy(target_spacings)
        self.filenames = []
        self.p_quantified  = []
        self.channels  = []
        if(target_db=="diffusion"):
            filename = './data/diffusion_%d.mat' % (seed)
            # Preload file and serialize
            contents = hdf5storage.loadmat(filename)
            channels = np.asarray(contents['output_h'], dtype=np.complex64)
            # Use only first subcarrier of each symbol
            self.channels.append(channels[:, 0])
            # Convert to array
            self.channels = np.asarray(self.channels)
            self.channels = np.reshape(self.channels,
                   (-1, self.channels.shape[-2], self.channels.shape[-1]))
        elif (target_db=="cgan"):
            filename = './data/cgan.mat'
            contents = hdf5storage.loadmat(filename)
            self.channels = np.asarray(contents['output_da'], dtype=np.float64)
            self.p_quantified = np.asarray(contents['input_da'], dtype=np.float64)
            
        # Normalize
        if type(norm) == list:
            self.mean = norm[0]
            self.std  = norm[1]
        elif norm == 'entrywise':
            self.mean = np.mean(self.channels, axis=0)
            self.std  = np.std(self.channels, axis=0)
        elif norm == 'global':
            self.mean = 0.
            self.std  = np.std(self.channels)
            
        # Generate random QPSK pilots
        if(target_db == "diffusion"):
            self.pilots = 1/np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(
                self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1 + \
                    1j * (2 * np.random.binomial(1, 0.5, size=(
                self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1))
        elif(target_db=="cgan"):
            last_dim_data = [
                1 + 0j, 0.9749 + 0.2225j, 0.9010 + 0.4339j, 0.7818 + 0.6235j,
                0.6235 + 0.7818j, 0.4339 + 0.9010j, 0.2225 + 0.9749j, 0 + 1j
                ]
        
            self.pilots = np.tile(last_dim_data, (self.channels.shape[0], config.data.image_size[0], 1))
            
            t_channels = torch.from_numpy(self.channels)
            real = t_channels[:,:,:,0]  
            imaginary = t_channels[:,:,:,1]  
            complex_numbers = torch.complex(real, imaginary)
            self.channels = complex_numbers.numpy()
    
            t_p_quantified = torch.from_numpy(self.p_quantified)
            t_real = t_p_quantified[:,:,:,0]  
            t_imaginary = t_p_quantified[:,:,:,1]  
            t_complex_numbers = torch.complex(t_real, t_imaginary)
            t_complex_numbers = torch.transpose(t_complex_numbers, -1, -2)
            self.p_quantified = t_complex_numbers.numpy()
        
        # Complex noise power
        self.noise_power = 1/np.sqrt(2) * config.data.noise_std
            
    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Normalize
        H_cplx = self.channels[idx]
        H_cplx_norm = (H_cplx - self.mean) / self.std
        
        # Convert to reals
        H_real_norm = \
            np.stack((np.real(H_cplx_norm), np.imag(H_cplx_norm)), axis=0)
        
        # Get complex pilots and create noisy Y
        P = self.pilots[idx]
        Y = np.matmul(H_cplx, P)
        N = self.noise_power * (np.random.normal(size=Y.shape) + \
                                1j * np.random.normal(size=Y.shape))
        Y = Y + N
        if len(self.p_quantified) ==0 :
            Y_quantified = Y
        else:
            Y_quantified = self.p_quantified[idx]
            
        # Compute largest eigenvalue of normal operator
        eigvals = np.real(
            np.linalg.eigvals(np.matmul(
                P, np.conj(P.T))))
        
        # Also get Hermitian H, real-viewed
        H_herm      = np.conj(np.transpose(H_cplx))
        H_herm_norm = np.conj(np.transpose(H_cplx_norm))
        H_real_herm_norm = \
            np.stack((np.real(H_herm_norm), np.imag(H_herm_norm)), axis=0)

        # And more Hermitians
        P_herm = np.conj(np.transpose(P))
        Y_herm = np.conj(np.transpose(Y))

        sample = {'H': H_real_norm.astype(np.float32),
                  'H_herm': H_real_herm_norm.astype(np.float32),
                  'H_herm_cplx': H_herm.astype(np.complex64),
                  'P': self.pilots[idx].astype(np.complex64),
                  'P_herm': P_herm.astype(np.complex64),
                  'Y': Y.astype(np.complex64),
                  'Y_herm': Y_herm.astype(np.complex64),
                  'eig1': eigvals[0].astype(np.float32),
                  'sigma_n': self.noise_power.astype(np.float32),
                  'Y_quantified': Y_quantified.astype(np.complex64),
                  'idx': int(idx)}
        return sample