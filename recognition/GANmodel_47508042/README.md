# Pattern Recognition - Generative Model using VQVAE

## Overview
This project implements a Vector Quantised Variational Autoencoder (VQVAE) to learna generative model from the HipMRI
prostate cancer study dataset. THe model reconstructs 2D MRI slices using compressed discrete latent codes, effectively
learning meaningful anatomical representation of prostate structures. The objective of this is to produce clear, realistic
reconstructions of MRI slices with a SSIM > 0.6 on testing.

## How it works
VQVAE is a discrete latent variable model that combines that representational power of neural networks with the 
compression efficiency of vector quantization. It consists of three main components:
1. Encoder: A convolutional neural network that maps input MRI slices to a continuous latent space.
2. Vector Quantizer: This module discretizes the continuous latent representations into a finite set of learned embedding vectors (codebook).
3. Decoder: Another convolutional neural network that reconstructs the MRI slices from the quantized latent codes.
The model is trained end-to-end using a combination of reconstruction loss (Mean Squared Error) and a commitment loss (VQ loss)

## Files
 - 'dataset.py' - Encoder, Decoder, VectorQuantizer, VQVAE model
 - 'modules.py' - 'HipMRIDataset' loader
 - 'predict.py' - load best model and produce reconstructions
 - 'train.py' -  training loop, validation, checkpointing, plots
 - 'utils.py' - helper functions for training and evaluation
 - 'readme.md' - documentation

## Install:
```bash
pip install torch torchvision matplotlib scikit-image numpy nibabel tqdm
```

## Training
To train the model, run:
```bash
python3 train.py --data_root "/Users/justin/Downloads/HipMRIDataset"
```

## Prediction
To generate reconstructions using the best model, run:
```bash
python3 predict.py --checkpoint outputs/best_checkpoint.pth --data_root "/Users/justin/Downloads/HipMRIDataset"
```

## Author
Justin Teng (47508042)