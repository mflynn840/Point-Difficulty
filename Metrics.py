
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

class RunningVOG:

    def __init__(self, shape):
        self.shape = shape
        self.n = 0
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)
    
    '''
    
        data is of the form (num samples, gradient dimension)
    
    '''
    def update(self, data):
        self.n += 1
        delta = data-self.mean
        self.mean += delta/self.n
        delta2 = data-self.mean
        self.m2 += delta*delta2
    
    def get_mean(self):
        return self.mean

    def get_varience(self):
        if self.n >1:
            return self.m2 / (self.n-1)
    
    def get_VOGs(self, labels):
        varience = self.get_varience()
        VOG = (1/self.shape[1]) * torch.sum(varience, 1)
        
        #vog[i] is score for point i, labels[i] is the class point i belongs to
        class_means = []
        class_sds = []
        
        labs = np.unique(labels)
        for cls in np.unique(labs):
            class_scores = VOG[labels==cls]
            class_means.append(torch.mean(class_scores))
            class_sds.append(torch.std(class_scores))

        class_means = torch.tensor(class_means)
        class_sds = torch.tensor(class_sds)
        
        normalized_VOG = torch.empty_like(VOG)
        for i in range(len(VOG)):
            cls_index = labs.tolist().index(labels[i].item())  # Get index of the class
            mean = class_means[cls_index]
            sd = class_sds[cls_index]
            
            # Normalize using (VOG - mean) / std
            normalized_VOG[i] = (VOG[i] - mean) / sd if sd > 0 else 0  # Avoid division by zero

        return VOG
    
    def get_slice_VOGs(self, slice_idxs):
        all_vogs = self.get_VOGs()
        
        slice_VOGS = []
        for i in range(len(slice_idxs)):
            slice_VOGS.append(torch.mean(all_vogs[slice_idxs[i]]))
            
        return torch.stack(slice_VOGS)


'''given a list of scores for each datapoint, compute the scores for each slice as the average'''
def convert_to_slice_score(point_scores, slices):

    scores = torch.zeros((len(slices)))
    
    for i, slice_idx in enumerate(slices):
        scores[i] = torch.mean(point_scores[slice_idx])
    return scores


'''
Compute EL2N scores for each datapoint

EL2N(xi, yi) = || model(xi) - yi ||_2

where yi is a one hot encoding of xi's ground truth
'''
def point_EL2N(model, device, dataset):
    X = dataset.X.to(device)
    Y = dataset.Y.to(device)
    logits = model(X)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    Y_one_hot = torch.nn.functional.one_hot(Y, num_classes=logits.size(1)).float()
    score = torch.norm(probabilities - Y_one_hot, p=2, dim=1)
    return score.detach().cpu()



def slice_EL2N(model, device, dataset, slices):
    point_scores = point_EL2N(model, device, dataset)
    return convert_to_slice_score(point_scores, slices)

'''
Compute GRAND scores for each datapoint
'''
def point_GRAND(model, device, dataset):
    model.zero_grad()
    X = dataset.X.to(device)
    Y = dataset.Y.to(device)
    
    
    no_reduction_loss = torch.nn.CrossEntropyLoss(reduction='none')

    output_layer = model.model[-1]
    output_layer_name = list(model.named_modules())[-1][0]
    
    
    #turn off gradients for all layers except the last
    for name, param in model.named_parameters():
        if output_layer_name in name and "weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
            
    
    logits = model(X)
    loss = no_reduction_loss(logits, Y)
    
    scores = torch.zeros((X.shape[0]))
    
    #run each datapoint through the model and get the gradient of the output layer weights w.r.t the output
    for i in range(X.shape[0]):
        loss[i].backward(retain_graph=True)
        cur_grad = output_layer.weight.grad.detach()
        scores[i] = torch.norm(cur_grad[Y[i]], p=2)
        model.zero_grad()
        
    for name, param in model.named_parameters():
        param.requires_grad = True
            
    return scores.cpu()

def slice_GRAND(model, device, dataset, slices):
    point_scores = point_GRAND(model, device, dataset)
    return convert_to_slice_score(point_scores, slices)
    
    
'''compute the loss for each datapoint in the dataset'''
def point_loss(model, device, dataset):
    X = dataset.X.to(device)
    Y = dataset.Y.to(device)
    logits = model(X)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    return criterion(logits, Y).detach().cpu()


def slice_loss(model, device, dataset, slices):
    point_scores = point_loss(model, device, dataset)
    return convert_to_slice_score(point_scores, slices)


'''since auc-roc is not defined per-point compute the scores for each slice.  if the score cannot be computed then score it as 0'''
def slice_auc_roc(model, device, dataset, slices):
    y_true = dataset.Y.numpy()
    y_pred = torch.softmax(model(dataset.X.to(device)), dim=1)[:,1].detach().cpu().numpy()


    scores = torch.zeros((len(slices)))
    for i, slice_idx in enumerate(slices):
        if len(np.unique(y_true[slice_idx])) > 1:
            scores[i] = roc_auc_score(y_true[slice_idx], y_pred[slice_idx])
        else:
            print("could not AUC compute for slice " + str(i) + "because it only has 1 class in it")
    return scores

'''since accuracy is not defined per point, compute it per slice'''
def slice_accuracy(model, device, dataset, slices):
    y_true = dataset.Y.numpy()
    y_pred = torch.argmax(model(dataset.X.to(device)), dim=1).detach().cpu().numpy()
    

    scores = torch.zeros((len(slices)))
    for i, slice_idx in enumerate(slices):
        scores[i] = accuracy_score(y_true[slice_idx], y_pred[slice_idx])

    return scores





