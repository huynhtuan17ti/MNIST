import torch
import torchvision
from torchvision import transforms
from models import CNN_model
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np 
import os
import warnings
import argparse
warnings.filterwarnings("ignore")

BATCH_SIZE = 150
NUM_WORKERS = 0
random_seed = 2020
LR = 0.001
EPOCHS = 6
log_interval = 90
save = True

train_losses = []
train_counter = []
test_losses = []
test_counter = []

def init():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', default = 150, help='batch size')
    ap.add_argument('--lr', default = 0.001, help='learning rate')
    ap.add_argument('--epochs', default = 6, help='number of epochs')
    ap.add_argument('--log_interval', default = 90, help='number of epochs before each evaluation')
    ap.add_argument('--save', default = True, help = 'save history figure True|False')

    args = ap.parse_args()

    global BATCH_SIZE, LR, EPOCHS, log_interval, save
    BATCH_SIZE = int(args.batch_size)
    LR = float(args.lr)
    EPOCHS = int(args.epochs)
    log_interval = int(args.log_interval)
    save = bool(args.save)

    

def train(net, optimizer, epoch, train_loader):
    net.train()
    loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to('cuda')
        target = target.to('cuda')
        optimizer.zero_grad()
        output = net(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset),
                                                                     100.*batch_idx/len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*BATCH_SIZE + (epoch-1)*len(train_loader.dataset))
            )
            torch.save(net.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test(net, test_loader):
    net.eval()
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to('cuda')
            target = target.to('cuda')
            output = net(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                                                                        100. * correct/len(test_loader.dataset)))


def plot_history():
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = 'blue')
    plt.scatter(test_counter, test_losses, color = 'red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('categorical crossentropy loss')
    plt.savefig(os.path.join('./results', 'history.png'))

if __name__ == '__main__':

    torch.backends.cudnn.enabled = False
    torch.manual_seed = random_seed
    
    init()

    train_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_loader = DataLoader(MNIST('./data/', train = True, download = True, transform = train_transforms), 
                                    batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)

    test_loader = DataLoader(MNIST('./data/', train = False, download = True, transform = test_transforms), 
                                    batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)

    net = CNN_model()
    net = net.to('cuda')
    optimizer = optim.Adam(net.parameters(), lr = LR)

    test_counter = [i*len(train_loader.dataset) for i in range(EPOCHS+1)]
    print('Starting training ...............')
    test(net, test_loader)
    
    for epoch in range(1, EPOCHS+1):
        train(net, optimizer, epoch, train_loader)
        test(net, test_loader)
    if save:
        print('Saving history figure !')
        plot_history()

