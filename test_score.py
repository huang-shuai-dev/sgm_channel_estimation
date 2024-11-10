#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from loaders              import Channels
from torch.utils.data     import DataLoader
from matplotlib import pyplot as plt
from datetime import datetime

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--quant',action='store_true')
parser.add_argument('--debug',action='store_true')
parser.add_argument('--highsnr',action='store_true')
parser.add_argument('--nocondscore',action='store_true')
parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--test', type=str, default='CDL-C')
parser.add_argument('--db', type=str, default='diffusion')
#TODO @shahaung why use this, what the function
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
#TODO @shahaung why use this, what the function
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.6])
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Target file
target_dir  = './models/score/%s' % args.db
target_file = os.path.join(target_dir, 'final_model.pt')
contents    = torch.load(target_file)
config      = contents['config']
now = datetime.now()
# Default hyper-parameters for pilot_alpha = 0.6, all SNR points
if args.train == 'CDL-A':
    # !!! Not to be confused with 'pilot_alpha' that denotes fraction of pilots
    alpha_step = 3e-11 # 'alpha' in paper Algorithm 1
    beta_noise = 0.01  # 'beta' in paper Algorithm 1
elif args.train == 'CDL-B':
    alpha_step = 3e-11
    beta_noise = 0.01
elif args.train == 'CDL-C':
    alpha_step = 3e-11
    beta_noise = 0.01
elif args.train == 'CDL-D':
    alpha_step = 3e-11
    beta_noise = 0.01
elif args.train == 'Mixed':
    alpha_step = 3e-11
    beta_noise = 0.01
# Number of Langevin steps at each noise level
config.sampling.steps_each = 3

# Instantiate model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
# Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

# Train and validation seeds
train_seed, val_seed = 1234, 4321
# Get training dataset for normalization
config.data.channel = args.train
config.db           = args.db
dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Range of SNR, test channels and hyper-parameters
if args.highsnr:
    snr_range          = np.arange(10, 12.5, 2.5)
else:
    snr_range          = np.arange(-10, 32.5, 2.5)
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
pilot_alpha_range  = np.asarray(args.pilot_alpha)
noise_range        = 10 ** (-snr_range / 10.) * config.data.image_size[1]
# Number of validation channels
num_channels = 100
    
# Global results
now      = datetime.now()
datetime_string = now.strftime('%Y_%m_%d_%H_%M_%S')
nmse_log = np.zeros((len(spacing_range), len(pilot_alpha_range),
                     len(snr_range), int(config.model.num_classes * \
                   config.sampling.steps_each), num_channels))
result_dir = './results/score/test_db_%s_%s' % (
    args.db,datetime_string)
os.makedirs(result_dir, exist_ok=True)
if args.debug:
    debug_file = open(os.path.join(result_dir, 'debug'),"w")
    debug_value_1=[]
    debug_value_2=[]


# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, pilot_alpha_range)
                    
# For each hyper-combo
for meta_idx, (spacing, pilot_alpha) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range)))
    
    # Get validation dataset
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.test
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots   = int(np.floor(config.data.image_size[1] * pilot_alpha))
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=num_channels,
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader)
    print('There are %d validation channels' % len(val_dataset))
        
    # Get all validation pilots and channels
    val_sample = next(val_iter)
    val_P = val_sample['P'].cuda()
    # Hermitian pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm'].cuda()
    val_Y_quant = val_sample['Y_quantified'].cuda()
    val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    # Initial estimates
    init_val_H = torch.randn_like(val_H)
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        # Get received pilots at correct SNR
        # We directly sample unit power complex-valued tensors via torch.randn_like
        # This is correct but partially undocumented as of PyTorch 2.1 - see https://github.com/pytorch/pytorch/issues/118269 for details
        
        val_Y     = torch.matmul(val_P, val_H)
        val_Y     = val_Y + \
            np.sqrt(local_noise) * torch.randn_like(val_Y)
        
        current   = init_val_H.clone()
        
        if(args.db=="diffusion"):
            y         = val_Y
            if(args.quant):
                # take quant: 
                Y_real = torch.real(y)
                Y_imag = torch.imag(y)
        
                # cal the mean of the image and real part
                mean_real = torch.mean(Y_real)
                mean_imag = torch.mean(Y_imag)
                q_up_real =   torch.max(Y_real) - mean_real
                q_dn_real =   mean_real - torch.min(Y_real)
                q_up_imag =   torch.max(Y_imag) - mean_imag
                q_dn_imag =   mean_imag - torch.min(Y_imag)
                
                q_max = max(q_up_real,q_dn_real,q_up_imag,q_dn_imag)

                if args.debug:
                        print( "QMAX",q_up_real,q_dn_real,q_up_imag,q_dn_imag, \
                                         file=debug_file)
        
                #quant for the image and real part
                real_mask = y.real > mean_real
                y.real = torch.where(real_mask, torch.ones_like(y.real), -torch.ones_like(y.real))
        
                imag_mask = y.imag > mean_imag
                y.imag = torch.where(imag_mask, torch.ones_like(y.imag), -torch.ones_like(y.imag))
        elif(args.db=="cgan"):
            # cgan db has quantified already
            q_max     = 0.5
            y         = val_Y_quant

        #TODO shahuang check the forward and forward_h: what's the function and why don't use quantified value
        forward   = val_P
        forward_h = torch.conj(torch.transpose(val_P, -1, -2))
        norm      = [0., 1.]
        oracle    = val_H # Ground truth channels
        # Count every step
        trailing_idx = 0
        
        for step_idx in tqdm(range(val_config.model.num_classes)):
            # Compute current step size and noise power
            current_sigma = diffuser.sigmas[step_idx].item()
            # Labels for diffusion model
            labels = torch.ones(init_val_H.shape[0]).cuda() * step_idx
            labels = labels.long()
            
            # Compute annealed step size
            alpha = alpha_step * \
                (current_sigma / val_config.model.sigma_end) ** 2
            
            # For each step spent at that noise level
            for inner_idx in range(val_config.sampling.steps_each):
                
                # Compute score using real view of data
                current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                with torch.no_grad():
                    score = diffuser(current_real, labels)
                # View as complex
                score = \
                    torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                
                # Compute gradient for measurements in un-normalized space
                meas_grad = torch.matmul(forward_h, 
                             torch.matmul(forward, current) - y)
                mean = torch.matmul(forward_h,torch.matmul(forward, current))
                # Sample noise
                grad_noise = (np.sqrt(2 * alpha * beta_noise) ) * \
                    torch.randn_like(current) 
                
                # Apply update
                if(args.quant or args.db=="cgan"):
                    cond_score = meas_grad /(local_noise/2. + current_sigma ** 2 + ( ((q_max)**2)/12))
                    current = current + alpha * (score - cond_score ) + grad_noise
                elif(args.nocondscore):
                    cond_score = meas_grad /(local_noise/2. + current_sigma ** 2 )
                    current = current + alpha * (score ) + grad_noise
                else:
                    cond_score = meas_grad /(local_noise/2. + current_sigma ** 2 )
                    current = current + alpha * (score - cond_score ) + grad_noise
                
                if args.debug:
                    print(current.mean(),score.mean(),meas_grad.mean(),grad_noise.mean(), cond_score.mean(),\
                                     local_noise,current_sigma,file=debug_file)
                    debug_value_1.append(torch.norm(score.mean()).item())
                    debug_value_2.append(torch.norm(cond_score.mean()).item())
                        
                # Store loss
                nmse_log[spacing_idx, pilot_alpha_idx, snr_idx, trailing_idx] = \
                    (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2))/\
                    torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()
                trailing_idx = trailing_idx + 1
if args.debug:
    plt.figure(figsize=(10, 10))
    plt.plot(debug_value_1,'g')
    plt.plot(debug_value_2,'r')
    plt.savefig(os.path.join(result_dir, 'results_debug.png'), dpi=300, 
            bbox_inches='tight')
    plt.figure(figsize=(10, 10))
    plt.plot(debug_value_2,'r')
    plt.savefig(os.path.join(result_dir, 'results_debug_condition.png'), dpi=300, 
            bbox_inches='tight')
# Use average estimation error to select best number of steps
avg_nmse  = np.mean(nmse_log, axis=-1)
best_nmse = np.min(avg_nmse, axis=-1)

# Plot results for all alpha values
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 10))
for alpha_idx, local_alpha in enumerate(pilot_alpha_range):
    plt.plot(snr_range, 10*np.log10(best_nmse[0, alpha_idx]),
             linewidth=4, label='Alpha=%.2f' % local_alpha,marker='*')
plt.grid(); plt.legend()
plt.title('Score-based channel estimation')
plt.xlabel('SNR [dB]'); plt.ylabel('NMSE [dB]')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results.png'), dpi=300, 
            bbox_inches='tight')
plt.close() 
               
# Save results to file based on noise
save_dict = {'nmse_log': nmse_log,
             'avg_nmse': avg_nmse,
             'best_nmse': best_nmse,
             'spacing_range': spacing_range,
             'pilot_alpha_range': pilot_alpha_range,
             'snr_range': snr_range,
             'val_config': val_config,
            }
torch.save(save_dict, os.path.join(result_dir, 'results.pt'))
if args.debug:
    debug_file.close()