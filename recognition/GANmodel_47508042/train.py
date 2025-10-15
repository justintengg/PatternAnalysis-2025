import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from modules import VQVAE

def train_loop(args):
    device = torch.device('cuda' if torch.cude.is_available() else 'cpu')
    print("device:", device)
