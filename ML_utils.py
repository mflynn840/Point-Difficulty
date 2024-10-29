import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import pickle as pkl
import os
import numpy as np



def get_slice_idx_list(path="./slice.idx"):
    with open(path, 'rb') as file:
        return pkl.load(file)
        
        

'''get gradients vog needs to compute VOG for each datapoint'''
def get_VOG_grads(model, device, dataset):
    model.zero_grad()
    X = dataset.X.to(device)
    Y = dataset.Y.to(device)
    X.requires_grad_(True)
    logits = model(X)
    logits = logits[:, Y]
    logits.backward(torch.ones_like(logits))
    input_grad = X.grad
    S = input_grad.detach().cpu()
    return S

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
        layers.append(nn.LeakyReLU(.01))

        #hiden layers
        for i in range(n_hidden+1):
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
        
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, metrics_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_dict = metrics_dict

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Call the parent method to save the model
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Prepare the file path for metrics
        metrics_path = os.path.join(self.dirpath, f"{self.filename}.metrics")

        # Save the metrics dictionary as a JSON file
        with open(metrics_path, 'wb') as f:
            pkl.dump(self.metrics_dict, f)
            
           