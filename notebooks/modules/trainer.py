import argparse
import numpy as np
import sys
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import utils
import models
import pickle


DATA_DIR = os.path.join('../..', 'datasets', 'FMNIST_DATASET')
BATCH_SIZE = 200
SEED = 42
LOAD_MODEL = False
LR = 1e-4
EPOCHS = 200
BETA = 1e-3
MODEL_SAVE_PATH = os.path.join('..', 'saved_models', 'fmnist_gmm_vae.h5')
PICKLE_SAVE_PATH = os.path.join('..', 'pickles', 'gmm_vae.pkl')
K = 64
W = 4


def main(raw_args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    parser = argparse.ArgumentParser(description='Train a VIB or VAE')
    parser.add_argument('-bla', type=str, nargs='?', help='Bla', default='bla')
    args = parser.parse_args(raw_args)

    if os.path.isdir(DATA_DIR):
        try:
            train_data = FashionMNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor())
            test_data = FashionMNIST(root=DATA_DIR, train=False, transform=transforms.ToTensor())
        except:
            train_data = FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
            test_data = FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    else:
        os.mkdir(DATA_DIR)
        train_data = FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
        test_data = FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True)

    test_loader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False)

    FMNIST_CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gmm_model = models.FMNIST_GMM_VAE(K, W, device).to(device)
    gmm_model = utils.train_model_vae(EPOCHS, gmm_model, train_loader,
                                      test_loader, device, LR, MODEL_SAVE_PATH, BETA)
    utils.pickle_model_stats(gmm_model, PICKLE_SAVE_PATH)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))



