import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import argparse

def load_mnist(path = './data/unzip', s = 'train'):
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

    print("Load {} mnist data successfully!".format(s))
    print("{}: {} images".format(s, len(images)))
    return images, labels

if __name__ == '__main__':
    load_mnist()

        