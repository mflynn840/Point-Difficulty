import pickle as pkl
from utils import AdultDataset, correlation_matrix
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import os

'''data is of the form (slice, num_scores)'''
def bar_chart(data, bar_names, epoch, xlabs=np.arange(128), xlab="slice idx", ylab="normalized score"):

    nbars = len(bar_names)
    # Set up the x locations for the bars
    x = np.arange(len(xlabs))

    # Set the width of the bars
    bar_width = 0.05

    
    # Create the bar chart
    for i in range(nbars):
        bars = plt.bar(x + i * bar_width, data[:, i], width=bar_width, label=bar_names[i])

        # Add names above each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, bar_names[i], 
                    ha='center', va='bottom')
    # Add labels and title
    plt.xlabel('Randomly selected slice indicies scored at epoch ' + str(epoch))
    plt.ylabel('rank')
    plt.title('Scoring functions at epoch' + str(epoch))
    plt.xticks(x + bar_width * (nbars - 1) / 2, xlabs)  # Set x-ticks to be centered

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    

def get_metrics(path="./checkpoints/34298864.metrics"):
    with open(path, 'rb') as file:
        return pkl.load(file)

def visualize_pca(data):
    # Perform PCA
    pca = PCA(n_components=2)  # You can extract more components if needed
    principal_components = pca.fit_transform(data)

    # Extract PCA1
    pca1 = principal_components[:, 0]  # First principal component
    pca2 = principal_components[:, 1]  # Second principal component (optional)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(pca1, pca2, c='blue', edgecolor='k', s=50)
    plt.title('PCA Visualization')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid()
    plt.axhline(0, color='red', linewidth=0.8, linestyle='--')  # Add horizontal line at y=0
    plt.axvline(0, color='red', linewidth=0.8, linestyle='--')  # Add vertical line at x=0
    plt.show()

    
    
def var_plot(vars, score_names, num_slices, epoch):
    # Plotting the variance for each metric
    for i, metric in enumerate(score_names):
        plt.figure(figsize=(8, 5))
        plt.bar(range(num_slices), vars[metric], color='skyblue')
        plt.xticks(range(num_slices), [f'Slice {j+1}' for j in range(num_slices)])
        plt.title(f'Variance of {metric} Across Slices for ' + metric + " on epoch " + str(epoch))
        plt.xlabel('Slices')
        plt.ylabel('Variance')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def score_slice(point_scores, slice_idx):
    return torch.mean(point_scores[slice_idx])

def parse_metrics(epoch, random_slices):

    bar_names = ["loss", "accuracy", "AUC-ROC", "GRAND", "EL2N", "PCA1"]

    lower_is_better = {
        "loss" : True,
        "accuracy" : False,
        "AUC-ROC" : False,
        "GRAND" : True,
        "EL2N" : True,
        "VOG" : True
    }
    

    all_metrics = {}
    i=0
    for metname in os.listdir("./checkpoints"):
        if ".metrics" in metname:
            fname = os.path.join("./checkpoints", metname)
            with open(fname, 'rb') as file:
                all_metrics[i] = pkl.load(file)
                i += 1
                
    avg_scores = {}  # Assuming 5 slices


    for metric_idx, metric in enumerate(bar_names):
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

    # Apply PCA
    pca = PCA(n_components=1)  # Reduce to 2 dimensions
    principal_components = pca.fit_transform(scores)
    
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





#metrics = get_metrics("./checkpoints/62459821.metrics")
random_slices = np.random.randint(0, high=128, size=(7), dtype='l')

parse_metrics(20, random_slices)
#bar_chart()   
    