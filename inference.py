import torch
import torchvision
from torchvision import transforms
from models import CNN_model, FaissKNeighbors, Average_Samples
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np 
import os
import warnings
import argparse
from custom_datasets import load_mnist
warnings.filterwarnings("ignore")

N_NEIGHBORS = 1
BATCH_SIZE = 150
NUM_WORKERS = 0
arith_ratio = 100
save = True

def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_neighbors', default = 1, help='number of neighbors')
    ap.add_argument('--save', default = True, help = 'save result figure True|False')
    ap.add_argument('--arith_ratio', default = 100, help = 'arithmetical ratio for saving accuracy in prediction (must > 0)')

    args = ap.parse_args()
    global N_NEIGHBORS, save, arith_ratio
    N_NEIGHBORS = int(args.n_neighbors)
    arith_ratio = int(args.arith_ratio)
    save = bool(args.save)

def inference_CNN():
    acc_CNN = []

    print('-'*70)
    print("\tStarting predict in test set with CNN model !")
    print('-'*70)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_loader = DataLoader(MNIST('./data/', train = False, download = True, transform = test_transforms), 
                                    batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = False)
    print('Loading test set successfully !')
    net = CNN_model()
    net.load_state_dict(torch.load('./results/model.pth'))
    net = net.to('cuda')
    print('Loading model sucessfully!')

    net.eval()
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to('cuda')
            target = target.to('cuda')
            output = net(data)
            pred = output.data.max(1, keepdim = True)[1]
            correcr_arr = pred.eq(target.data.view_as(pred))
            for j in range(len(correcr_arr)):
                correct += correcr_arr[i].item()
                if (i*BATCH_SIZE + j + 1)%arith_ratio == 0:
                    acc_CNN.append(100. * correct/(i*BATCH_SIZE + j + 1))

    print('\nCNN model: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                                        100. * correct/len(test_loader.dataset)))
    del net
    torch.cuda.empty_cache()
    return acc_CNN
    
def inference_Average_Samples():
    print('-'*70)
    print("\tStarting predict in test set with Average_Samples !")
    print('-'*70)
    train, target = load_mnist()
    model = Average_Samples()
    model.fit(train, target)
    print('Loading model successfully!\n')
    
    test, target = load_mnist(s = 't10k')
    correct, acc_AS = model.score(test, target, arith_ratio = arith_ratio)

    print('\nAverage_Samples model: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(target),
                                                                        100. * correct/len(target)))
    return acc_AS


def inference_KNN():
    print('-'*70)
    print("\tStarting predict in test set with {} nearest neighbors !".format(N_NEIGHBORS))
    print('-'*70)

    train, target = load_mnist()
    nsamples, nx, ny = train.shape
    d2_train = train.reshape((nsamples, nx*ny))
    model = FaissKNeighbors(k=N_NEIGHBORS)
    model.fit(d2_train, target)
    print('Loading model successfully!\n')
    test, target = load_mnist(s = 't10k')
    nsamples, nx, ny = test.shape
    d2_test = test.reshape((nsamples, nx*ny))
    predictions = model.predict(d2_test)
    correct, acc_KNN = model.score(predictions, target, arith_ratio = arith_ratio)
    print('\nKNN model: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(target),
                                                                        100. * correct/len(target)))
    return acc_KNN

def plot_result(acc_CNN, acc_KNN, acc_AS):
    fig = plt.figure()
    plt.plot(np.arange(arith_ratio, 10001, arith_ratio), acc_CNN, color = 'blue')
    plt.plot(np.arange(arith_ratio, 10001, arith_ratio), acc_KNN, color = 'red')
    plt.plot(np.arange(arith_ratio, 10001, arith_ratio), acc_AS, color = 'orange')
    plt.legend(['Avg. Accuracy CNN', 'Avg. Accuracy KNN', 'Avg. Accuracy Average Samples'], loc = 'upper right')
    plt.xlabel('number of test examples')
    plt.ylabel('Avg. Accuracy')
    plt.savefig(os.path.join('./results', 'result_acc.png'))

if __name__ == '__main__':
    init()
    acc_CNN = inference_CNN()
    acc_KNN = inference_KNN()
    acc_AS =  inference_Average_Samples()
    if save:
        plot_result(acc_CNN, acc_KNN, acc_AS)