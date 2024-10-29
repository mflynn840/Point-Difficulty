import pickle as pkl
import torch
import numpy as np
from sklearn.decomposition import PCA
import os




def get_metrics(path="./checkpoints/34298864.metrics"):
    with open(path, 'rb') as file:
        return pkl.load(file)


def parse_metrics(epoch, random_slices):

    bar_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "PCA1"]

    #need to flip scores that run in different directions
    lower_is_better = {
        "loss" : True,
        "accuracy" : False,
        "AUC-ROC" : False,
        "GRAND" : True,
        "EL2N" : True,
        "VOG" : True
    }
    

    #agregate metrics from the various runs
    all_metrics = {}
    i=0
    for metname in os.listdir("./checkpoints"):
        if ".metrics" in metname:
            fname = os.path.join("./checkpoints", metname)
            with open(fname, 'rb') as file:
                all_metrics[i] = pkl.load(file)
                i += 1
    
    
    #get the mean of each metric      
    avg_scores = {} 
    
    #for each metric
    for metric_idx, metric in enumerate(bar_names):
        
        #compute the average score of each slice for that metric
        avg = torch.zeros(128)
        count = 0
        for run in range(i):
            if (metric == "VOG" or metric == "PCA1"):
                continue
            if run != 2:
                print(run)
                scores = all_metrics[run][epoch][metric]
                avg += scores
                count += 1
            
        if not(metric == "VOG" or metric == "PCA1") and run != 2:
            avg /= count
            avg_scores[metric] = avg

    scores = np.zeros((128, 6))

    for j, (metric) in enumerate(bar_names):
        if j == 5:
            break
        scores[:, j] = avg_scores[metric].numpy()
        
        if lower_is_better.get(metric, True):
            scores[:, j] = -scores[:, j]


    for i in range()
    scores
    # Apply PCA
    pca = PCA(n_components=1)  # Reduce to 2 dimensions
    principal_components = pca.fit_transform(scores[:])
    #Normalize the scores with (Min-max normalization)
    
    
    for i in range(128):
        scores[i][5] = principal_components[i]


    vars = {}
    
    for metric_idx, metric in enumerate(bar_names):
        if metric != "PCA1":
            var = torch.zeros(128)
            for run in range(8):
                if epoch in all_metrics[run].keys():
                    var += (all_metrics[run][epoch][metric] - avg_scores[metric])**2
            
            var = (1/8) * var
            vars[metric] = var
        
    var_plot(vars,  ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N"], 128, epoch)

    
    ranks = np.argsort(np.argsort(-scores, axis=0), axis=0) + 1  # +1 to start ranks at 1
    correlation_matrix(ranks, bar_names, epoch)
    
    
    bar_chart(ranks[random_slices, :], bar_names, epoch, random_slices)
    visualize_pca(scores[:,:5])





if __name__ == "__main__":
    random_slices = np.random.randint(0, high=128, size=(7), dtype='l')
    parse_metrics(20, random_slices)

    