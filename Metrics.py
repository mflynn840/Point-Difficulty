from abc import ABC, abstractmethod
import torch


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
    
    def get_VOGs(self):
        varience = self.get_varience()
        VOG = (1/self.shape[1]) * torch.sum(varience, 1)
        return VOG
    
    def get_slice_VOGs(self, slice_idxs):
        all_vogs = self.get_VOGs()
        
        slice_VOGS = []
        for i in range(len(slice_idxs)):
            slice_VOGS.append(torch.mean(all_vogs[slice_idxs[i]]))
            
        return torch.stack(slice_VOGS)



class SliceLoss:
    
    def __init__(self):
        
            
class EL2N:

class GRAND:

class EL2N:

class PCA1:
    

