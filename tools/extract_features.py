import matplotlib.pyplot as plt
from matplotlib import gridspec 
import os
import numpy as np
import gzip
import argparse
import sys
sys.path.append('/home/ryan/Machine_Learning/MNIST/')
from custom_datasets import load_mnist
import tqdm
import time


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

def vectorize(X):
    print("Vectorize data set successfully {}x{}x{} to {}x{}".format(X.shape[0], X.shape[1], X.shape[2], X.shape[0], X.shape[1]*X.shape[1]))
    return X.reshape((X.shape[0], X.shape[1]*X.shape[2]))

def get_histogram(img, plot = False):
    hist = np.zeros((256))
    unique, counts = np.unique(img, return_counts = True)
    for x, y in zip(unique, counts):
        hist[int(x)] = int(y)
    if plot:
        return unique, counts
    return hist

def histogram(X):
    print("\nHistogram:")
    result = []
    process_bar = tqdm.tqdm(range(len(X)))
    for i, img in zip(process_bar, X):
        hist = get_histogram(img)
        result.append(hist)
    result = np.array(result)
    print("Convert data set to histogram successfully {}x{}x{} to {}x{}".format(X.shape[0], X.shape[1], X.shape[2], result.shape[0], result.shape[1]))
    return result

def get_downsampling(img, size, type):
    new_img = np.zeros((img.shape[0]//size, img.shape[1]//size))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j] = type(img[i*size : i*size + size, j*size : j*size + size])
    return new_img

def downsampling(X, func):
    print("\nDownsampling:")
    result = []
    process_bar = tqdm.tqdm(range(len(X)))
    for i, img in zip(process_bar, X):
        new_img = get_downsampling(img, 2, func)
        result.append(new_img)
    result = np.array(result)
    print("Downsampling data set successfully {}x{}x{} to {}x{}x{}".format(X.shape[0], X.shape[1], X.shape[2], 
                                                                        result.shape[0], result.shape[1], result.shape[2]))
    return result

def average_matrix(matrix):
    return matrix.mean()

def max_matrix(matrix):
    return np.amax(matrix)

def min_matrix(matrix):
    return np.amin(matrix)

def plot_histogram(X_train, y_train):
    num = np.random.randint(0, 9)
    img = X_train[y_train == num][0]
    x, y = get_histogram(img, plot=True)
    fig = plt.figure(figsize=(12, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 6], height_ratios=[6, 1])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(img, cmap ='Greys', interpolation = 'nearest')
    ax0.set_title("Raw image", fontsize = 9)

    ax1 = plt.subplot(gs[1])
    ax1.bar(x, y, align = 'center', facecolor = 'b')
    ax1.set_title("Histogram", fontsize = 15)
    ax1.set_ylim(0, max(y)+5)
    ax1.set_ylabel("Counts", fontsize = 10)
    ax1.set_xlabel("Pixels", fontsize = 10)
    ax1.grid(True)
    plt.tight_layout(pad = 6.0)
    plt.savefig(os.path.join(SAVE_PATH, 'examples_histogram.png'))
    print("Save histogram examples image successfully!")


def plot_downsampling(X_train, y_train, func):
    fig, ax = plt.subplots(figsize=(12,18), nrows=5, ncols=4)
    ax = ax.ravel()

    fig.suptitle("Examples of downsampling (get {})".format(TYPE_FUNC), y = 1, fontsize = 20)

    num = 0
    for i in range(0, 20, 2):
        img = X_train[y_train == num][0]
        ax[i].imshow(img, cmap ='Greys', interpolation = 'nearest')
        ax[i].set_title("Raw image ({}x{})".format(img.shape[0], img.shape[1]), fontsize = 12)
        new_img = get_downsampling(img, 2, func)
        ax[i+1].imshow(new_img, cmap = 'Greys', interpolation = 'nearest')
        ax[i+1].set_title("DownSampling ({}x{})".format(new_img.shape[0], new_img.shape[1]), fontsize = 12)
        num += 1
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout(pad = 3.0)
    plt.savefig(os.path.join(SAVE_PATH, 'examples_downsampling_{}.png'.format(TYPE_FUNC)))
    print("Save downsampling examples image successfully!")
    

if __name__ == '__main__':
    init()

    func = average_matrix
    if TYPE_FUNC == "max":
        func = max_matrix
    if TYPE_FUNC == "min":
        func = min_matrix
    if TYPE_FUNC != "max" and TYPE_FUNC != "min" and TYPE_FUNC != "average":
        print("Not found suitable function, using default average function!")

    X_train, y_train = load_mnist(DATA_PATH)
    vectorize(X_train)
    #histogram(X_train)
    #downsampling(X_train, func)

    #Save examples image in results folder
    print('-'*60)
    plot_downsampling(X_train, y_train, func)
    plot_histogram(X_train, y_train)
    
