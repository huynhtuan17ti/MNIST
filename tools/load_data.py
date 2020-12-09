import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import argparse
import sys
sys.path.append('/home/ryan/Machine_Learning/MNIST/')
from custom_datasets import load_mnist

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='./data/unzip',
                                help = 'path to data folder')
    ap.add_argument('--save', default='./results',
                                help = 'path to save result')
    args = ap.parse_args()

    data_path = args.path
    X_train, y_train = load_mnist(data_path)
    print('Rows: {}, columns: {}'.format(X_train.shape[0], X_train.shape[1]))

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0]
        ax[i].imshow(img, cmap ='Greys', interpolation = 'nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    #save img
    save_path = args.save
    plt.savefig(os.path.join(save_path, 'samples.png'))
    plt.show()