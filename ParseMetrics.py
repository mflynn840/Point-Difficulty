import pickle as pkl
from utils import AdultDataset, correlation_matrix
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

'''data is of the form (slice, num_scores)'''
def bar_chart(data, bar_names, epoch, xlabs=np.arange(128), xlab="slice idx", ylab="normalized score", nbars=4):


    # Set up the x locations for the bars
    x = np.arange(len(xlabs))

    # Set the width of the bars
    bar_width = 0.15

    
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
    plt.title('Scoring functions')
    plt.xticks(x + bar_width * (nbars - 1) / 2, xlabs)  # Set x-ticks to be centered

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    

def get_metrics(path="./checkpoints/34298864.metrics"):
    with open(path, 'rb') as file:
        return pkl.load(file)
        
    

    

def score_slice(point_scores, slice_idx):
    return torch.mean(point_scores[slice_idx])

def parse_metrics(metrics, slice_idx_list, epoch):

    bar_names = ["loss", "GRAND", "EL2N", "VOG", "PCA1"]

    random_slices = np.random.randint(0, high=128, size=(10), dtype='l')
    print(metrics.keys())
    
    for epoch in [3, 6, 18]:
        #scores[i][j] is slice is score using scoring function j
        scores = np.zeros((128, 5))
        for i, slice in enumerate(slice_idx_list):
            cur_scores = np.zeros((5))
            for j, metric in enumerate(bar_names):
                if j == 4:
                    break
                score = score_slice(metrics[epoch][metric], slice)
                cur_scores[j] = score
            
            scores[i] = cur_scores
        
        
        # Apply PCA
        pca = PCA(n_components=1)  # Reduce to 2 dimensions
        principal_components = pca.fit_transform(scores)
        
        for i in range(128):
            scores[i][4] = principal_components[i]
        

        
        #print(principal_components)
        #print(scores[:,3])
        ranks = np.argsort(np.argsort(-scores, axis=0), axis=0) + 1  # +1 to start ranks at 1
        correlation_matrix(ranks, bar_names, epoch)
        
        
        bar_chart(ranks[random_slices, :], bar_names, epoch, random_slices, nbars=5)
        
    




metrics = get_metrics()
idxs = get_slice_idx_list()
parse_metrics(metrics, idxs, 9)
#bar_chart()   
    