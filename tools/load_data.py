import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import argparse

def load_mnist(path, s = 'train'):
    labels_path = os.path.join(path, "{}-labels-idx1-ubyte.gz".format(s))
    images_path = os.path.join(path, "{}-images-idx3-ubyte.gz".format(s))

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype = np.uint8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28, 28).astype(np.float64)

    return images, labels

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='F:\\Bai Tap\\Nhap mon CNTT - TH\\MNIST\\data',
                                help = 'path to data folder')
    ap.add_argument('--save', default='F:\\Bai Tap\\Nhap mon CNTT - TH\\MNIST\\result',
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
    plt.savefig(os.path.join(save_path, 'result.png'))
    plt.show()