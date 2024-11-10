# MIMO Channel Estimation using Score-Based (Diffusion) Generative Models


## Requirements
Python 3.8, 3.9, or 3.10. Tested on Ubuntu 20.04 and 22.04. MATLAB license required to run MATLAB scripts.

## Getting Started
After cloning the repository, run the following commands for Python 3.10 (similar for other versions of Python):
- `cd score-based-channels`
- `python3.10 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

This will create a self-contained virtual environment in the base directory, activate it, and install all required packages.

### Pre-generated Data
There are two dataset: 
The dataset from cGAN based one-bit ADC channel estimation.
The dataset from the score-based-estimation.
Both of them are under the data directory

### Pre-trained Models
A pre-trained diffusion model is under the model/score for both databases.


## Training Diffusion Models on MIMO Channels
Train the diffusion model with diffusion database
```
python train_score.py
```
Train the diffusion model with cGAN database
```
python train_score.py --db cgan
```

The last model weights will be automatically saved in the `model` folder under the appropriate structure. 

## Channel Estimation with Diffusion Models
To run channel estimation with the pretrained model:
With diffusion database
```
python test_score.py
````
If you want add quantization behavior for the received pilot signal
```
python test_score.py --quant
````
With cGAN database
```
python test_score.py --db cgan
````

