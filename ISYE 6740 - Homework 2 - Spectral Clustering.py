# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:30:43 2020

@author: Graham Billey
"""
import time
# import os
#from os.path import abspath, exists
import pandas as pd
import numpy as np
# from scipy import sparse
from matplotlib import pyplot as plt

'''
0. Rewrite K-Means functions from Homework 1
1. Read and clean the data.
2. Create the Laplacian matrix.
3. Calculate the eigen decomposition and sort based on smallest eigenvalues.
4. Run k-means on the 2 smallest eigenvalues to classify the data.
'''

##################################################################
# Rewrite K-Means Code from HW 1
##################################################################

# Step 1. initialize the means.
def initialize_means(data, K, rand_seed=1):
    np.random.seed(rand_seed)
    means = np.random.choice(len(data), replace=False, size=K)
    return data[means, :]

# Step 2. Calculate distances from each point to the mean of each cluster
def calculate_distances_to_means(data, means, l_norm=2):
    num_points = len(data)
    K = len(means)
    dists = np.zeros((num_points, K))
    for i in range(num_points):
        dist = np.linalg.norm(data[i,:] - means, axis=1, ord=l_norm)
        dists[i,:] = dist**l_norm
    return dists

# Step 3. Assign each data point to a cluster based on the distances calculated in step 3.
def assign_each_point_to_closest_mean(dists):
    return np.argmin(dists, axis=1)
    
# Step 4. Update the means based on the fixed cluster assignments.
def update_means(data, means, l_norm=2):
    new_means = np.zeros(means.shape)
    distances = calculate_distances_to_means(data, means, l_norm)
    labels = assign_each_point_to_closest_mean(distances)
    
    for cluster in set(labels):
        points_in_cluster = np.array([data[j] for j in range(len(data)) if labels[j] == cluster])
        new_means[cluster] = np.mean(points_in_cluster, axis=0)

    return new_means

# Step 5. 
def check_for_convergence(means, new_means):
    return np.array_equal(means, new_means)

# Putting it all together.
def k_means(data, K, max_iterations = 10, l_norm = 2):
    print('Running k_means...')
    t0 = time.time()
    converged = False
    means = initialize_means(data, K)
    iteration = 1
    
    while (not converged) and (iteration < max_iterations):

        distances = calculate_distances_to_means(data, means, l_norm)
        labels = assign_each_point_to_closest_mean(distances)
        new_means = update_means(data, means)
        
        converged = check_for_convergence(means, new_means)
        means = new_means
        iteration += 1

    t1 = time.time()
    runtime = round(t1-t0)
    
    return labels, means, iteration, runtime
        
##################################################################
# End of K-Means code
##################################################################
    
k = 3

''' Read the data '''
nodes = pd.read_csv('nodes.txt', sep='\t', names=['index', 'url', 'label', 'category'])
nodes['index'] = nodes['index'] - 1
edges = pd.read_csv('edges.txt', sep='\t', names=['from', 'to'])
edges['from'] = edges['from'] - 1
edges['to'] = edges['to'] - 1

''' Clean the data '''
# Build adjacency matrix
A = np.zeros(shape=(len(nodes),len(nodes)))
for index, row in edges.iterrows():
    f = row['from']
    t = row['to']
    if f != t:
        A[f,t] = 1
        A[t,f] = 1

# Find indices of all nodes that are connected to other nodes
A_pd = pd.DataFrame(A)
A_pd['total_connections'] = A_pd.sum(axis=1)
nodes_to_keep = A_pd.index[A_pd['total_connections'] != 0].tolist()

# Filter the adjacency matrix and convert back to numpy array
A_filtered = A_pd.iloc[nodes_to_keep,nodes_to_keep]
A_f = A_filtered.values
#A_sparse = sparse.coo_matrix(A_f)

# Remove the nodes that had no connections as well
nodes_pd = pd.DataFrame(nodes)
nodes_f = nodes.iloc[nodes_to_keep]


''' Calculate the Symmetric normalized Laplacian matrix '''
D = np.diag(np.sum(A_f, axis=1))
D_norm = np.diag(1/np.sqrt(np.sum(A_f, axis=1)))
L = D - A_f
L_sym = D_norm @ L @ D_norm

''' Calculate the eigen decomposition and sort based on smallest eigenvalues '''
values, vectors = np.linalg.eigh(L_sym)
vectors = vectors.real
vectors_k = vectors[:, 0:k]
vectors_k_norm = vectors_k/np.repeat(np.sqrt(np.sum(vectors_k*vectors_k, axis=1).reshape(-1, 1)), k, axis=1)

''' Visualize the eigenvectors '''
# plt.scatter(vectors_k[:, 0], vectors_k[:, 1])
# plt.show()
# np.where(values < 0.1)[0]
# vectors[0]
# vectors[1]
# plt.scatter(values, range(len(values)))

''' Run K-Means on the eigenvector representation of the data '''

labels, means, iteration, runtime = k_means(data=vectors_k_norm, K=k, max_iterations=200)
labels_pd = pd.DataFrame(labels, columns={'label'})
labels_pd.groupby(by='label').size()

''' Compare the predicted labels to the true labels '''
nodes_f['pred'] = labels
# nodes_f.head()
nodes_agg = nodes_f.groupby(by=['label', 'pred']).size().reset_index(name='count').sort_values(by='count', ascending=False)
print(nodes_agg)

correct = 0
labels = list(set(nodes_agg['label']))
pred_groups = list(set(nodes_agg['pred']))
for index, row in nodes_agg.iterrows():
    check = row['label'] in (labels) and row['pred'] in (pred_groups)
#     print(check)
    if check:
        correct += row['count']
        try:
#             print('label ',int(row['label']))
            labels.pop(int(row['label']))
        except:
            pass
        try:
#             print('pred ',int(row['pred']))
            pred_groups.pop(int(row['pred']))
        except:
            pass

grouping_accuracy = correct/sum(nodes_agg['count'])
print(f'Model accurately grouped {correct} nodes.')
print(f'Model achieved {round(grouping_accuracy*100,1)} % accuracy!\n')
print(f'The false classification rate is {round((1-grouping_accuracy)*100,1)} %\n')


