import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset
import pickle as pkl
import torch.functional as F


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class AdultDataset(Dataset):
    def __init__(self, path):
        self.X, self.Y = self.parseData(path)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def parseData(self, f_name):
        with open(f_name, 'rb') as file:
            return pkl.load(file)
        
        

def get_slice_ga_errors2(model, X, Y, slice_idxs):
        device = "cuda:0"
        

        model.eval()
        model.zero_grad()
        
        model.zero_grad()
        X = X.to(device)
        Y = Y.to(device, non_blocking=True)
        X.requires_grad_(True)
        logits = model(X)
        logits = logits[:, Y]
        logits.backward(torch.ones_like(logits))
        input_grad = X.grad
        point_grads = input_grad.detach().cpu()


        avg_dset_grad = torch.mean(point_grads, dim=0)

        GA_scores = []
        for i in range(len(slice_idxs)):
            GA_score = torch.norm(avg_dset_grad - torch.mean(point_grads[slice_idxs[i]], dim=0), p=2)
            GA_scores.append(GA_score)
            
        GA_scores = torch.stack(GA_scores)
            
        print(GA_scores.shape)
        return GA_scores
    
    
class SimpleNN(nn.Module):
    
    def __init__(self, in_features, n_hidden, layer_size, n_classes):
        super(SimpleNN, self).__init__()
        
        self.n_layers = n_hidden + 2
        layers = []
        
        #input layer
        layers.append(nn.Linear(in_features, layer_size))

        #hiden layers
        for i in range(n_hidden-1):
            layers.append(nn.Linear(layer_size, layer_size)) 
            layers.append(nn.LeakyReLU(0.01))

        #output layer
        layers.append(nn.Linear(layer_size, n_classes))
        
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)
        

        
    def forward(self, x):
        return self.model(x)
            

class AdultDataset(Dataset):
    def __init__(self, path):
        self.X, self.Y = self.parseData(path)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def parseData(self, f_name):
        with open(f_name, 'rb') as file:
            return pkl.load(file)
            
            
    
