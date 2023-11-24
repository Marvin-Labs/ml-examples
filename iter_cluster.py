"""
This script tackles prevalent challenges in clustering methodologies, focusing on three specific issues:

1. The frequent occurrence of large sets of unlabeled points.
2. Significant variation in cluster sizes, often leading to inconsistent grouping.
3. Wide disparities between the maximum and minimum sizes of clusters, which can impact the interpretability and usability of clustering results.

To address these issues, the script runs models with multiple hyparam combinations (within the `iter_cluster_gridsearch` function) on the dataset. For each combination, we iteratively recluster the unlabeled set until the remaining set of unlabeled points is reduced to a certain fraction of the original size. This is implemented by the `iter_cluster` function iteratively running the `cluster_cycle` function. This iteration is the key contribution in this script and it addresses challenge 1.

Secondly, to address challenges 2 and 3, the process generates a statistics dataframe which provides metadata about cluster distributions under each hyparam combination. This allows the user to study the effect of each hyparam on the cluster distribution statistics (number of clusters, mean cluster size, largest and smallest cluster, and cluster size standard deviation), and so tune in on the a hyparam combination that leads to a cluster distribution that suits the desired application.

Lastly, the script executes an ensemble clustering technique. This technique consolidates the results from all hyparam combinations, deriving a final clustering consensus, which makes sure that the final clusters are robust.

Currently, our clustering pipeline consists of UMAP for dimensionality reduction, followed by clustering with the HDBSCAN model. This is a frequently used approach and it was chosen for its simplicity, speed, and effectiveness. Of course, this can be replaced with any other clustering techinque if desired by the user.

We are currently working on adding a hierarchical clustering step that takes place after the completion of each hyparam combination's resulting model. This further stage aims to refine the clustering process, enhancing the granularity and accuracy of the results.
"""

import argparse
from copy import deepcopy
from hdbscan import HDBSCAN
from umap import UMAP
import torch
import torch.optim as optim
import random
from tqdm import tqdm
import gzip
import numpy as np
from pathlib import Path
from os.path import join as pathjoin
import requests
import json
from itertools import chain
import re
from random import shuffle
import pandas as pd
import math
import os
import scipy
import numpy as np
import itertools
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import faiss
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def linspace_without_duplicates(*args, **kwargs):
    """
    Create a linearly spaced array with no duplicates.

    Args:
    *args, **kwargs: Arguments for numpy's linspace function.

    Returns:
    List of unique values from generated linspace.
    """
    linspace = np.linspace(*args, **kwargs)
    return list(set(linspace))

def prettify(param):
    """
    Format a parameter for display, truncating floats to 4 decimal places.

    Args:
    param (float or any): Parameter to format.

    Returns:
    Formatted parameter.
    """
    return f'{param:.4f}' if isinstance(param, float) else param

def label_stats(labels, display=True, unlabels=1):
    """
    Calculate and optionally display statistics of clustering labels.

    Args:
    labels (array-like): Array of cluster labels.
    display (bool): If True, prints the statistics.
    unlabels (int): Index to start counting labels from.

    Returns:
    List of statistics [total, outliers, max_size, min_size, mean, std].
    """
    counts = np.array([(labels == i).sum() for i in np.unique(labels)])[unlabels:]
    total = len(counts)-unlabels
    outliers = (labels == -1).sum()
    max_size = int(np.max(counts))
    min_size = int(np.min(counts))
    mean = counts.mean()
    std = counts.std()
    if display:
        print('Final label count stats:')
        print('#labels:', len(counts)-unlabels)
        print('#outliers:', (labels == -1).sum())
        print('max:', int(np.max(counts)))
        print('min:', int(np.min(counts)))
        print('mean:', f'{counts.mean():.2f}')
        print('std:', f'{counts.std():.2f}')
    return [total, outliers, max_size, min_size, mean, std]


#
# def label_stats(labels, display=True, unlabels=1):
#
#     # Unlabels: usually it is just the -1 label which stands for unlabeled.
#
#     counts = np.array([(labels == i).sum() for i in np.unique(labels)])[unlabels:]
#     total = len(counts)-unlabels
#     outliers = (labels == -1).sum()
#     max_size = int(np.max(counts))
#     min_size = int(np.min(counts))
#     mean = counts.mean()
#     std = counts.std()
#     if display:
#         print('Final label count stats:')
#         print('#labels:', len(counts)-unlabels)
#         print('#outliers:', (labels == -1).sum())
#         print('max:', int(np.max(counts)))
#         print('min:', int(np.min(counts)))
#         print('mean:', f'{counts.mean():.2f}')
#         print('std:', f'{counts.std():.2f}')
#     return [total, outliers, max_size, min_size, mean, std]
#
# def reduce(vecs,
#             n_components =5 ,
#             n_neighbors=100,
#           metric='cosine',
#           min_dist=.1,
#           spread = 1,):
#     reducer = UMAP(n_neighbors=n_neighbors,n_components=n_components,
#                    metric=metric,spread=spread,min_dist=min_dist)
#     reduced = reducer.fit_transform(vecs)
#     return reduced

def cluster_cycle(vecs,
                  n_components =5 ,
                  n_neighbors=100,
          metric='cosine',
          min_dist=.1,
          spread = 1,
          cluster_selection_epsilon = .2, alpha = .7,
          cluster_selection_method = 'leaf', prediction_data = True,
          min_cluster_size=40,
          max_cluster_size=400):
    """
    Perform clustering on vectors using UMAP for dimensionality reduction and HDBSCAN for clustering.

    Args:
    vecs (array-like): Input data for clustering.
    n_components (int): Number of dimensions for UMAP reduction.
    n_neighbors (int): Number of neighbors for UMAP.
    metric (str): Metric for UMAP.
    min_dist (float): Minimum distance parameter for UMAP.
    spread (float): Spread parameter for UMAP.
    cluster_selection_epsilon (float): Epsilon parameter for HDBSCAN clustering.
    alpha (float): Alpha parameter for HDBSCAN.
    cluster_selection_method (str): Cluster selection method for HDBSCAN.
    prediction_data (bool): If True, HDBSCAN uses prediction data.
    min_cluster_size (int): Minimum cluster size for HDBSCAN.
    max_cluster_size (int): Maximum cluster size for HDBSCAN.

    Returns:
    tuple: A tuple (labels, reduced), where 'labels' are the cluster labels, and 'reduced' is the reduced dataset.
    """
    # Ensure the cluster selection epsilon is accurately formatted to 4 decimal places.
    # This might help avoid numerical precision issues during clustering.
    cluster_selection_epsilon = float(f'{float(cluster_selection_epsilon):.4f}')

    # Initialize UMAP reducer with the specified parameters. UMAP is used for reducing
    # the dimensionality of the data, making it easier to cluster.
    reducer = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                   metric=metric, spread=spread, min_dist=min_dist)

    print('Reducing...')
    # Apply the UMAP reduction to the input vectors.
    reduced = reducer.fit_transform(vecs)

    # Initialize the HDBSCAN clustering algorithm with specified parameters.
    # HDBSCAN is used for clustering the reduced data.
    clustering = HDBSCAN(min_cluster_size=min_cluster_size,
                         max_cluster_size=max_cluster_size,
                         metric='euclidean',  # using Euclidean distance for clustering
                         cluster_selection_epsilon=cluster_selection_epsilon,
                         alpha=alpha,
                         cluster_selection_method=cluster_selection_method,
                         prediction_data=prediction_data)

    print('Fitting...')
    # Apply the HDBSCAN clustering to the reduced data.
    clustering.fit(reduced)

    # Retrieve the cluster labels assigned by HDBSCAN.
    labels = clustering.labels_

    # Print the number of unique clusters found (excluding noise points).
    print(f'{np.unique(labels).shape[0] - 1} clusters')

    # Print the count of points in each cluster.
    print('Label counts:', end=' ')
    for i in np.unique(labels):
        print((labels == i).sum(), end=' ')

    # Return the cluster labels and the reduced data.
    return labels, reduced


def iter_cluster(vecs,
                 min_cluster_size=5,
                 max_cluster_size=20,
                 n_components=5,
                 n_neighbors=20,
                 metric='mannhattan',
                 min_dist=1,
                 spread=1.25,
                 cluster_selection_epsilon=.0,
                 alpha=.1,
                 cluster_selection_method='eom',
                 prediction_data=False,
                 stop_frac = 10):
    """
    Iteratively cluster a dataset using UMAP and HDBSCAN with the specified parameters.

    Args:
    vecs (array-like or torch.Tensor): Input data for clustering.
    min_cluster_size (int): Minimum size of clusters for HDBSCAN.
    max_cluster_size (int): Maximum size of clusters for HDBSCAN.
    n_components (int): Number of dimensions for UMAP.
    n_neighbors (int): Number of neighbors for UMAP.
    metric (str): Metric for UMAP (e.g., 'manhattan').
    min_dist (float): Minimum distance parameter for UMAP.
    spread (float): Spread parameter for UMAP.
    cluster_selection_epsilon (float): Epsilon parameter for HDBSCAN.
    alpha (float): Alpha parameter for HDBSCAN.
    cluster_selection_method (str): Cluster selection method for HDBSCAN.
    prediction_data (bool): If True, use prediction data in HDBSCAN.
    stop_frac (int): Fraction of the dataset size used as a stopping criterion.

    Returns:
    array: Final cluster labels for each vector in the dataset.
    """
    # Ensure the cluster selection epsilon is accurately formatted to 4 decimal places.
    # This might help avoid numerical precision issues during clustering.
    cluster_selection_epsilon = float(f'{float(cluster_selection_epsilon):.4f}')
    if type(vecs) == torch.Tensor:
        vecs = vecs.numpy()

    # Initialize variables for the iterative clustering process
    N, DIM = vecs.shape
    leftovers = vecs
    final_labels = np.array([-1] * N)
    leftover_inds = np.arange(N)
    n_cycle = 0
    label_offset = 0
    n_clusters = 1

    # Iteratively perform clustering
    while len(leftovers) > N / stop_frac and n_clusters:
        print('Cycle:', n_cycle)

        # Perform clustering on the remaining data
        labels, _ = cluster_cycle(leftovers, min_cluster_size=min_cluster_size,
                                  max_cluster_size=max_cluster_size,
                                  n_components=n_components,
                                  n_neighbors=n_neighbors,
                                  metric=metric,
                                  min_dist=min_dist,
                                  spread=spread,
                                  cluster_selection_epsilon=cluster_selection_epsilon,
                                  alpha=alpha,
                                  cluster_selection_method=cluster_selection_method,
                                  prediction_data=prediction_data,
                                  )

        # Count the number of clusters excluding noise (-1 label)
        n_clusters = np.unique(labels).shape[0] - 1

        print('\n# clusters:', n_clusters)
        print('# leftovers:', (labels == -1).sum(),'\n')

        # Update cycle count and check for termination
        n_cycle += 1
        if not n_clusters:
            break

        # Update the arrays for the next iteration
        leftovers = leftovers[labels == -1]
        labeled_inds = leftover_inds[np.where(labels != -1)[0]]
        final_labels[leftover_inds] = np.array([l if l == -1 else l + label_offset for l in labels])
        leftover_inds = leftover_inds[np.where(labels == -1)[0]]
        label_offset += n_clusters

    # Print cluster distribution stats if there are more than 2 clusters (2 clusters is meaningless because one is the unlabeled set)
    if np.unique(final_labels).shape[0] > 2:
        label_stats(final_labels, display=True)

    return final_labels

def iter_cluster_gridsearch(vecs,dataset_name, n_combs=20, cache=False,
                            stop_frac=10, collapse_threshold_stds=3, hp_num=5,
                            min_cluster_size_min=5, min_cluster_size_max = 50,
                            max_cluster_size_min=20, max_cluster_size_max=500,
                            n_components_min=3, n_components_max=100,
                            n_neighbors_min=10, n_neighbors_max=100,
                            min_dist_min=0.01, min_dist_max=1.0,
                            spread_min=0.5, spread_max=2.0,
                            cluster_selection_epsilon_min=0.01, cluster_selection_epsilon_max=1.0,
                            alpha_min=0.01, alpha_max=1.0,
                            ):
    """
    Performs a grid search over a range of hyperparameters to cluster vectors using a specified algorithm.

    Parameters:
    - vecs (array-like): Vectors to be clustered.
    - dataset_name (str): Name of the dataset, used for naming output files.
    - n_combs (int, optional): Number of hyperparameter combinations to try. Default is 20.
    - cache (bool, optional): If True, results will be cached. Default is False.
    - stop_frac, collapse_threshold_stds, hp_num: Additional hyperparameters for the clustering algorithm.
    - min_cluster_size_min, min_cluster_size_max, max_cluster_size_min, max_cluster_size_max,
      n_components_min, n_components_max, n_neighbors_min, n_neighbors_max,
      min_dist_min, min_dist_max, spread_min, spread_max,
      cluster_selection_epsilon_min, cluster_selection_epsilon_max,
      alpha_min, alpha_max: Range parameters for various hyperparameters used in the clustering algorithm.

    Returns:
    - DataFrame: A pandas DataFrame containing the results of the grid search, including statistics about the clusters formed under each set of hyperparameters.

    Notes:
    - The function iteratively applies the clustering algorithm with different combinations of hyperparameters.
    - Results are optionally cached and saved to a CSV file for further analysis.
    - The function skips combinations where `min_dist` is greater than `spread`.
    - This function requires external dependencies like pandas, numpy, and itertools.

    Example Usage:
    ```python
    results_df = iter_cluster_gridsearch(my_vectors, 'my_dataset')
    ```
    """

    # Set the output path for the results and the cache directory based on the dataset name
    outpath = f'output/{dataset_name}-cluster-gridsearch.csv'
    cache_dir = f'output/{dataset_name}-labels'

    # Define the names of hyperparameters to be used in the grid search
    hp_names = ['min_cluster_size', 'max_cluster_size', 'n_components', 'n_neighbors', 'metric', 'min_dist',
                'spread', 'cluster_selection_epsilon', 'alpha', 'cluster_selection_method', 'prediction_data']

    # Generate choices for each hyperparameter, ensuring no duplicates within each range
    min_cluster_size_choices = linspace_without_duplicates(min_cluster_size_min, min_cluster_size_max, num=hp_num,
                                                           dtype=int)
    max_cluster_size_choices = linspace_without_duplicates(max_cluster_size_min, max_cluster_size_max, num=hp_num,
                                                           dtype=int)
    n_components_choices = linspace_without_duplicates(n_components_min, n_components_max, num=hp_num, dtype=int)
    n_neighbors_choices = linspace_without_duplicates(n_neighbors_min, n_neighbors_max, num=hp_num, dtype=int)
    metric_choices = ['cosine', 'euclidean', 'manhattan']
    min_dist_choices = linspace_without_duplicates(min_dist_min, min_dist_max, num=hp_num)
    spread_choices = linspace_without_duplicates(spread_min, spread_max, num=hp_num)
    cluster_selection_epsilon_choices = linspace_without_duplicates(cluster_selection_epsilon_min,
                                                                    cluster_selection_epsilon_max, num=hp_num)
    alpha_choices = linspace_without_duplicates(alpha_min, alpha_max, num=hp_num)
    cluster_selection_method_choices = ['leaf', 'eom']
    prediction_data_choices = [True, False]

    # Create all possible combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(
        min_cluster_size_choices, max_cluster_size_choices, n_components_choices, n_neighbors_choices, metric_choices,
        min_dist_choices, spread_choices, cluster_selection_epsilon_choices, alpha_choices,
        cluster_selection_method_choices, prediction_data_choices
    ))

    # Shuffle the hyperparameter combinations for randomness
    shuffle(hyperparameter_combinations)

    # Check if the output file already exists, and load its content if it does
    if os.path.exists(outpath):
        df = pd.read_csv(outpath, index_col=0)
        results = df.values.tolist()
    else:
        results = []

    # Create the output directories if they don't exist
    os.makedirs('output', exist_ok=True)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)

    # Iterate over each hyperparameter combination
    i = 0
    for combination in hyperparameter_combinations:
        # Unpack the hyperparameters from the combination
        min_cluster_size, max_cluster_size, n_components, n_neighbors, metric, min_dist, spread, cluster_selection_epsilon, alpha, cluster_selection_method, prediction_data = \
            combination

        # Skip the combination if min_dist is greater than spread
        if min_dist > spread:
            continue

        # Display the current combination and its hyperparameters
        print('=========================')
        print(f'Combination #{i}/{n_combs}')
        comb = [min_cluster_size, max_cluster_size, n_components, n_neighbors, metric, min_dist, spread,
                cluster_selection_epsilon, alpha, cluster_selection_method, prediction_data]
        hp_vals = [prettify(param) for param in comb]  # Format the parameters for readability
        param_string = ','.join([name + '=' + str(hp_val) for (name, hp_val) in zip(hp_names, hp_vals)])
        print('hyparams:', param_string)
        print('=========================')

        # Perform the clustering with the current set of hyperparameters
        labels = iter_cluster(vecs, min_cluster_size=min_cluster_size,
                              max_cluster_size=max_cluster_size,
                              n_components=n_components, n_neighbors=n_neighbors,
                              metric=metric, min_dist=min_dist, spread=spread,
                              cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha,
                              cluster_selection_method=cluster_selection_method,
                              prediction_data=prediction_data,
                              stop_frac=stop_frac)

        # Count the number of items in each cluster, ignoring noise (-1 labels)
        counts = np.array([(labels == i).sum() for i in np.unique(labels)])[1:]

        # Skip this result if no clusters are formed
        if len(counts) == 0:
            continue
        # Compile the results for this combination
        this_result = label_stats(labels, display=False, unlabels=1) + hp_vals

        results.append(this_result)

        # Save the results to a DataFrame and write it to a CSV file
        df = pd.DataFrame(results, columns=['n_clusters', 'n_outliers', 'max', 'min', 'mean', 'std'] + hp_names)
        df.to_csv(f'output/{dataset_name}-cluster-gridsearch.csv')

        # Cache the labels if caching is enabled
        if cache:
            np.save(f'{cache_dir}/{param_string}.npy', labels)

        # Increment the counter and check if the maximum number of combinations has been reached
        i += 1
        if i > n_combs:
            break

    # Return the DataFrame containing all the results
    return df

def ensemble_consensus(dataset_name):

    paths = os.listdir(f'output/{dataset_name}-labels')
    item_labels = [np.load(f'output/{dataset_name}-labels/{path}') for path in paths]

    # Step 1: Create Co-association Matrix
    co_association_matrix = np.zeros((N, N))
    for labels in item_labels:
        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    co_association_matrix[i][j] += 1

    # Normalize the co-association matrix
    co_association_matrix /= n_combs

    # Step 2: Apply Hierarchical Clustering to the Co-association Matrix

    # Try different thresholds and see how they affect the final result.
    # Higher threshold=less clusters. Roughly speaking, pick the threshold
    # that gives you the number of thresholds you want and with the highest silhouette score
    for t in [1,2,3,4,5,6,7]:
        Z = linkage(co_association_matrix, method='average')
        consensus_cluster_labels = fcluster(Z, t=t, criterion='distance')
        np.fill_diagonal(co_association_matrix, 0)
        silhouette_avg = silhouette_score(co_association_matrix, consensus_cluster_labels, metric='precomputed')

        print(t,(np.unique(consensus_cluster_labels).shape)[0], silhouette_avg)

def pipeline_example():
    N = 1000
    n_combs = 3
    D = 100
    dataset_name = 'example'
    vecs = torch.randn(N, D)

    # This yields the dataframe with metadata about cluster distribution

    clust_df = iter_cluster_gridsearch(vecs=vecs, dataset_name=dataset_name,
                                       n_combs=n_combs, cache=True)
    # You can use it to study the effect of different hyparams
    # on cluster distribution and revise hyparam combination choices accordingly
    # to fit your application.


    ensemble_consensus(dataset_name)

if __name__ == '__main__':
    pipeline_example()