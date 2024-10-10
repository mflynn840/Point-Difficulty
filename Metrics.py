from abc import ABC, abstractmethod
import torch
import numpy as np

class PointDifficultyMetric(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def scorePoint(point, *kwargs):
        pass
    
    
    
class SliceDifficultyMetric(ABC):
    def __init__(self, slice_idxs, sc_function):
        
        #slice_idx[i][j] the ith slices jth datapoints index
        self.slice_idx = slice_idxs
        self.point_score = sc_function
    
    def score(self):
        return torch.mean()


    def score_slice():
        x=1
        
        
        
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

        return normalized_VOG
    
    def get_slice_VOGs(self, slice_idxs):
        all_vogs = self.get_VOGs()
        
        slice_VOGS = []
        for i in range(len(slice_idxs)):
            slice_VOGS.append(torch.mean(all_vogs[slice_idxs[i]]))
            
        return torch.stack(slice_VOGS)



class SliceLoss:
    
    def __init__(self):
        x=1
        
            
class EL2N:
    x=1

class GRAND:
    x=1

class EL2N:
    x=1

class PCA1:
    x=1
    

