import pickle as pkl
import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd


def rank_normalize(scores_mat):
    df = pd.DataFrame(scores_mat)
    ranked_df = df.rank(method='average', axis=0)
    return ranked_df.to_numpy()


def quantile_normalize(scores_mat):
    df = pd.DataFrame(scores_mat)
    ranked = df.rank(method='average', axis=0)
    mean_values = df.stack().groupby(ranked.stack()).mean()
    normalized_df = ranked.stack().map(mean_values).unstack()
    return normalized_df.to_numpy()


def get_metrics(path="./checkpoints/34298864.metrics"):
    with open(path, 'rb') as file:
        return pkl.load(file)


'''
Returns a matrix A where
A[i][j] is slice i's score under scoring function j (averaged over weight initilizations and normalized)
'''
def get_avg_scores(epoch, normalization = "quantile"):

    metric_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "VOG"]

    #need to flip scores that run in different directions
    lower_is_better = {
        "loss" : True,
        "accuracy" : False,
        "AUC-ROC" : False,
        "GRAND" : True,
        "EL2N" : True,
        "VOG" : True
    }
    


    #all_metrics[i][j][k][l] is run i, epoch j, scoring function k and slice l's score
    all_metrics = {}
    
    for r_name in os.listdir("./checkpoints"):
        if ".metrics" in r_name:
            fname = os.path.join("./checkpoints", r_name)
            with open(fname, 'rb') as file:
                all_metrics[r_name] = pkl.load(file)
    
    #get the mean of each metric over runs on the desired epoch   
    avg_scores = {} 
    
    #for each metric
    for metric in metric_names:
        
        #compute the average score of each slice for that metric
        avg = torch.zeros(128)
        n_runs = 0
        
        for run in all_metrics.keys():
            scores = all_metrics[run][epoch][metric]
            avg += scores
            n_runs += 1
            
        avg /= n_runs
        avg_scores[metric] = avg


    #scores_mat[i][j] is slice i's score under scoring function j
    scores_mat = np.zeros((128, 6))

    #get matrix representation
    for j, (metric) in enumerate(metric_names):
        scores_mat[:, j] = avg_scores[metric].numpy()
        
        if lower_is_better.get(metric, True):
            scores_mat[:, j] = -1 * scores_mat[:, j]

    print(scores_mat.shape)
    #normalize all scores
    if normalization == "quantile":
        return quantile_normalize(scores_mat)
        
    if normalization == "rank":
        return rank_normalize(scores_mat)
    


if __name__ == "__main__":
    scores_mat = get_avg_scores(epoch=20, normalization="rank")
    #print(scores_mat.shape)
    print(np.unique(scores_mat[:, 0]))

    