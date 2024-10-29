import torch
import pickle as pkl
import torch.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


'''plot multiple lines on a graph'''
def plot(Xs, Ys, labels, xlab="x", ylab="y", title="Title", markLast=False, percent=False):
    if type(Xs) is list:
        for X, Y, label in zip(Xs, Ys, labels):
            plt.plot(X, Y, label=label, linestyle='-')
            
            if markLast:
                if percent:
                    plt.text( X[-1],Y[-1], f'Final = {Y[-1]*100:.2f} %', fontsize=14, ha='right', va='bottom')
                else:
                    plt.text( X[-1],Y[-1], f'Final = {Y[-1]:.3f} ', fontsize=14, ha='right', va='bottom')
    else:
        plt.plot(X, Y)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)


    # Add a legend, grid, and show the plot
    plt.legend(markerscale=1.5)
    plt.grid(True)
    plt.show()
    
    
'''display a pearsons R correlation matrix'''
def correlation_matrix(scores, score_names, epoch):
    # Create a DataFrame from the scores array
    df = pd.DataFrame(scores, columns=score_names)

    # Calculate the correlation matrix between scoring functions
    correlation_matrix = df.corr()

    # Display the correlation matrix
    print("Correlation Matrix:\n", correlation_matrix)

    # Optional: Visualize the correlation matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Scoring Functions epoch' + str(epoch))
    plt.show()
  

        
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
