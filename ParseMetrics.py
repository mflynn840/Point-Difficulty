import pickle as pkl
import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd

from utils import spearman_matrix, PCA_correlations, explained_variance
import matplotlib.pyplot as plt
import seaborn as sns

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
        "VOG" : False
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
        
        if epoch == 0 and metric == "VOG":
            continue
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
        
        if epoch == 0 and metric  == "VOG":
            continue
        scores_mat[:, j] = avg_scores[metric].numpy()
        
        if lower_is_better.get(metric, True):
            scores_mat[:, j] = -1 * scores_mat[:, j]

    if epoch == 0:
        scores_mat = scores_mat[:, :5]
    #normalize all scores
    if normalization == "quantile":
        return quantile_normalize(scores_mat)
        
    if normalization == "rank":
        return rank_normalize(scores_mat)
    
    
    

def scoring_correlations():
    metric_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "VOG"]
    metric_names_0 = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N"]
    scores_mat_20 = get_avg_scores(epoch=20, normalization="quantile")
    scores_mat_15 = get_avg_scores(epoch=15, normalization="quantile")
    scores_mat_10 = get_avg_scores(epoch=10, normalization="quantile" )
    scores_mat_5 =  get_avg_scores(epoch=5, normalization="quantile")
    scores_mat_0 = get_avg_scores(epoch=0, normalization="quantile") 
    
    spearman_matrix(scores_mat_0, metric_names_0, 0)
    spearman_matrix(scores_mat_5, metric_names, 5)
    spearman_matrix(scores_mat_10, metric_names, 10)
    spearman_matrix(scores_mat_15, metric_names, 15)
    spearman_matrix(scores_mat_20, metric_names, 20)


def PCA_analysis():
    metric_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "VOG"]
    metric_names_0 =  ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N"]
    
    scores_mat_0 = get_avg_scores(epoch=0, normalization="quantile")
    scores_mat_20 = get_avg_scores(epoch=20, normalization="quantile")
    
    pca = PCA(n_components=6)
    principle_components_20 = pca.fit_transform(scores_mat_20)
    explained_variance_20 = pca.explained_variance_ratio_

    
    pca = PCA(n_components=5)
    principle_components_0 = pca.fit_transform(scores_mat_0)
    explained_varience_0 = pca.explained_variance_ratio_
    
    PCA_correlations(principle_components_20, scores_mat_20, metric_names, 20)
    PCA_correlations(principle_components_0, scores_mat_0, metric_names_0, 0)
    explained_variance(explained_variance_20, 20)
    explained_variance(explained_varience_0, 0)
    
    
'''how much of the top-k sets of each score function are the same'''
def scoring_similarity(topk):
    metric_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "VOG", "PCA-1"]


    epochs = [0, 5, 10, 15, 20]
    epoch_top_ks = []
    for epoch in epochs:
        scores_mat = get_avg_scores(epoch=epoch, normalization="quantile")
        
        if epoch == 0:
            scores_mat = np.hstack((scores_mat, np.zeros((scores_mat.shape[0], 1))))
            
        pca = PCA(n_components=1)
        principle_components = pca.fit_transform(scores_mat)
        scores_mat = np.hstack((scores_mat, principle_components))
        sorted_idxs = np.argsort(scores_mat, axis=0)
        epoch_top_ks.append(sorted_idxs[0:topk, :])

    epoch_top_ks = np.stack(epoch_top_ks)
    match_counts = np.zeros((5, 7, 7))
    print(epoch_top_ks.shape)
    
    for i in range(5):
        scores = epoch_top_ks[i]
        for j in range(7):
            for k in range(j + 1, 7):


                matches = np.intersect1d(scores[:, j], scores[:, k]).size

                match_counts[i, j, k] += matches
                match_counts[i, k, j] += matches 
    
    for i in range(7):
        match_counts[:, i, i] = topk
        
    #print(match_counts)
    match_counts /= topk
    # Output the match counts
    print("Match counts between scoring functions:")
    #print(match_counts)
    #print(epoch_top_ks.shape)
    
    fig, axes = plt.subplots(5, 1, figsize=(8, 4 * 5))

    for i in range(5):
        ax = axes[i]
        sns.heatmap(match_counts[i], annot=True, fmt='f', cmap='YlGnBu', ax=ax,
                    xticklabels=metric_names,
                    yticklabels=metric_names)
        ax.set_title(f'Epoch {epochs[i]}')
        ax.set_xlabel('Scoring Function')
        ax.set_ylabel('Scoring Function')

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
'''how much does the same scoring functions ranks change throughout training'''
def scoring_integrity():
    
    metric_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "VOG"]
    epochs = [0, 5, 10, 15, 20]
    
    
    for metric_name in metric_names:
        epoch_scores = []
        metric_idx = metric_names.index(metric_name)
        for epoch in epochs:
            scores_mat = get_avg_scores(epoch=epoch, normalization="quantile")
            
            if epoch == 0:
                scores_mat = np.hstack((scores_mat, np.zeros((scores_mat.shape[0], 1))))

            epoch_scores.append(scores_mat[:, metric_idx])
    
        epoch_scores = np.asarray(epoch_scores).T

        df_scores = pd.DataFrame(epoch_scores)
        correlation_matrix = df_scores.corr("spearman").values

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                    xticklabels=[f'Epoch {i+1}' for i in range(5)], 
                    yticklabels=[f'Epoch {i+1}' for i in range(5)])
        plt.title('Correlation Between Epochs (' + str(metric_name) + ")")
        plt.xlabel('Epochs')
        plt.ylabel('Epochs')
        plt.show()

        
    
if __name__ == "__main__":
    #scoring_correlations()
    #scoring_similarity(10)
    scoring_integrity()
    
    

