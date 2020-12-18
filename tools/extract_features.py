import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import argparse
import sys
sys.path.append('/home/ryan/Machine_Learning/MNIST/')
from custom_datasets import load_mnist

DATA_PATH = "./data/unzip"
SAVE_PATH = "./results"
TYPE_FUNC = "average"

def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='./data/unzip',
                                help = 'path to data folder')
    ap.add_argument('--save_path', default='./results',
                                help = 'path to save result')
    ap.add_argument('--type', default='average', 
                                help = 'type of function to downsampling (average | max | min)')
    global DATA_PATH, SAVE_PATH, TYPE_FUNC
    args = ap.parse_args()
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    TYPE_FUNC = args.type

def vectorize(img):
    return img.reshape((img.shape[0], img.shape[1]))

def downSampling(img, size, type):
    new_img = np.zeros((img.shape[0]//size, img.shape[1]//size))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j] = type(img[i*size : i*size + size, j*size : j*size + size])
    return new_img

def average_matrix(matrix):
    return matrix.mean()

def max_matrix(matrix):
    return np.amax(matrix)

def min_matrix(matrix):
    return np.amin(matrix)

def plot_samples(X_train):
    fig, ax = plt.subplots(figsize=(8,12), nrows=5, ncols=4)
    ax = ax.ravel()

    func = average_matrix
    if TYPE_FUNC == "max":
        func = max_matrix
    if TYPE_FUNC == "min":
        func = min_matrix
    if TYPE_FUNC != "max" and TYPE_FUNC != "min" and TYPE_FUNC != "average":
        print("Not found suitable function, using default average function!")

    num = 0
    for i in range(0, 20, 2):
        img = X_train[y_train == num][0]
        ax[i].imshow(img, cmap ='Greys', interpolation = 'nearest')
        ax[i].set_title("Raw image", fontsize = 10)
        new_img = downSampling(img, 2, func)
        ax[i+1].imshow(new_img, cmap = 'Greys', interpolation = 'nearest')
        ax[i+1].set_title("DownSampling ({})".format(TYPE_FUNC), fontsize = 10)
        num += 1
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout(pad = 3.0)
    plt.savefig(os.path.join(SAVE_PATH, 'samples_downsampling_{}.png'.format(TYPE_FUNC)))
    plt.show()

if __name__ == '__main__':
    init()
    X_train, y_train = load_mnist(DATA_PATH)
    plot_samples(X_train)
    
