# Pattern Recognition - Generative Model using VQVAE

## Overview
This project uses VQVAE to create a generative model of the HipMRI Study on Prostate Cancer. The goal is to produce
reasonably clear images and reach SSIM > 0.6 on testing.

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