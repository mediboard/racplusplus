import sys
import os
import pybind11
# Append the parent directory of racplusplus package to system path
sys.path.append(os.path.join(os.path.abspath(__file__), "..", "..", "build"))
sys.path.append("/Users/porterhunley/repos/racplusplus/build")
import numpy as np
import racplusplus
import scipy as sp

print("\nSee how it performs with a connectivity matrix")

# Set up matrix by size and density
rows = 10000
cols = 10000
density = 0.2
np.random.seed(42)
# Number of ones and zeros
num_ones = int(rows * cols * density)

# Generate random indices for ones
one_indices = np.random.choice(rows * cols, num_ones, replace=False)

# Make sure both (i, j) and (j, i) indices exist
rows_indices, cols_indices = np.unravel_index(one_indices, (rows, cols))
all_indices = np.concatenate([rows_indices, cols_indices, cols_indices, rows_indices])
all_cols = np.concatenate([cols_indices, rows_indices, rows_indices, cols_indices])

# Create a boolean array for data
data = np.ones(len(all_indices), dtype=bool)

# Create the sparse symmetric matrix
symmetric_connectivity_matrix = sp.sparse.csc_matrix((data, (all_indices, all_cols)), shape=(rows, cols))
print("Done generating sparse unweighted connectivity matrix.\n")

max_merge_distance = .24
batch_size = 1000
no_processors = 1
test_matrix = np.random.random((rows, 768))

print("Running RAC from Python using numpy data matrix and scipy sparse csc connectivity matrix.")
labels = racplusplus.rac(test_matrix, max_merge_distance, symmetric_connectivity_matrix, batch_size, no_processors)
print(f"Point Cluster Assignments: {labels}")
