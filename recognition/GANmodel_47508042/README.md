# Pattern Recognition - Generative Model using VQVAE

## Overview
This project uses VQVAE to create a generative model of the HipMRI Study on Prostate Cancer. The goal is to produce
reasonably clear images and reach SSIM > 0.6 on testing.

## Files
 - 'dataset.py' - Encoder, Decoder, VectorQuantizer, VQVAE model
 - 'modules.py' - 'HipMRIDataset' loader
 - 'predict.py' - load best model and produce reconstructions
 - 'train.py' -  training loop, validation, checkpointing, plots

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- scikit-image
- numpy

Install:
```bash
pip install torch torchvision matplotlib scikit-image numpy