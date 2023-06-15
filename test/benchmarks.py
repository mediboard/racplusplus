import sys
import os
import pybind11
# Append the parent directory of racplusplus package to system path
sys.path.append(os.path.join(os.path.abspath(__file__), "..", "..", "build"))
sys.path.append("/Users/porterhunley/repos/racplusplus/build")
import numpy as np
import time
import racplusplus
import scipy as sp
from sklearn.cluster import AgglomerativeClustering

print("\nSee how it performs with a connectivity matrix")

# Set up matrix by size and density
rows = 10
cols = 10
density = .3 
np.random.seed(42)

num_ones = int(rows * cols * density)

# Generate random indices for ones
one_indices = np.random.choice(rows * cols, num_ones, replace=False)

# Make sure both (i, j) and (j, i) indices exist
rows_indices, cols_indices = np.unravel_index(one_indices, (rows, cols))
all_indices = np.concatenate([rows_indices, cols_indices, cols_indices, rows_indices])
all_indices = np.concatenate([all_indices, [rows-1]*rows, np.arange(rows)])

all_cols = np.concatenate([cols_indices, rows_indices, rows_indices, cols_indices])

# Include new cols indices for the connections from the last row to all other rows
all_cols = np.concatenate([all_cols, np.arange(rows), [rows-1]*rows])

# Create a boolean array for data
data = np.ones(len(all_indices), dtype=bool)

# Create the sparse symmetric matrix
symmetric_connectivity_matrix = sp.sparse.csc_matrix((data, (all_indices, all_cols)), shape=(rows, cols))
print("Done generating sparse unweighted connectivity matrix.\n")
print(symmetric_connectivity_matrix.todense())

max_merge_distance = .24
batch_size = 1000
no_processors = 1
test_matrix = np.random.random((rows, 768))

# print(np.dot(test_matrix, test_matrix.T) / (np.linalg.norm(test_matrix) g.norm(test_matrix)))

print("Running RAC from Python using numpy data matrix and scipy sparse csc connectivity matrix.")
labels = racplusplus.rac(test_matrix, max_merge_distance, symmetric_connectivity_matrix, batch_size, no_processors)
print(f"Point Cluster Assignments: {len(set(labels))}")

start = time.time()
clustering = AgglomerativeClustering(
    n_clusters=None, 
    linkage='average',
    distance_threshold=max_merge_distance, 
    connectivity=symmetric_connectivity_matrix,
    metric='cosine').fit(test_matrix)

print(f"Sklearn Point Cluster Assignments: {len(set(clustering.labels_))}")

rac_labels = labels
sklearn_labels = clustering.labels_

sklearn_label_map = {key:i for (i, key) in enumerate(sklearn_labels)}
rac_label_map = {key:i for (i, key) in enumerate(rac_labels)}
transformed_sklearn = [sklearn_label_map[i] for i in sklearn_labels]
trans_rac_labels = [rac_label_map[i] for i in rac_labels]

print(transformed_sklearn)
print(trans_rac_labels)

print(trans_rac_labels == transformed_sklearn)
  
end = time.time()
print(end - start)