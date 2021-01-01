import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np

class CNN_model(nn.Module):

    def __init__(self, p = 0.0):
        super(CNN_model, self).__init__()
        self.p = p
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d(p = p)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = self.p, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Average_Samples:
    def __init__(self):
        self.average_samples = []

    def fit(self, X, y):
        for Class in range(10):
            average_sample = np.zeros(X[0].shape)
            count = 0
            for idx, sample in enumerate(X):
                if y[idx] == Class:
                    average_sample += sample
                    count += 1
            average_sample /= count
            self.average_samples.append(average_sample)
    
    def euclidean_distance(self, x_1, x_2):
        x = (x_1 - x_2)**2
        return np.sum(x)**(1/2)


    def predict(self, sample, ground_truth):
        dist = []
        for Class in range(10):
            dist.append(self.euclidean_distance(sample, self.average_samples[Class]))
        dist = np.array(dist)
        return np.argmin(dist) == ground_truth

    def score(self, X, y, arith_ratio):
        correct = 0
        acc = []
        for idx, sample in enumerate(X):
            correct += self.predict(sample, y[idx])
            if (idx+1)%arith_ratio == 0:
                acc.append(100. * correct/(idx+1))
        return correct, acc


class FaissKNeighbors:
    def __init__(self, k=1):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
    
    def score(self, pred, target, arith_ratio):
        acc = []
        correct = 0
        for i, (x, y) in enumerate(zip(pred, target)):
            if x == y:
                correct += 1
            if (i+1)%arith_ratio == 0:
                acc.append(100. * correct/(i+1))
            
        return correct, acc


if __name__ == '__main__':
    net = CNN_model()
    net.eval()
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    
