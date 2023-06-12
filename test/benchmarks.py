import sys
import os
import pybind11
# Append the parent directory of racplusplus package to system path
sys.path.append(os.path.join(os.path.abspath(__file__), "..", "..", "build"))
sys.path.append("/Users/porterhunley/repos/racplusplus/build")
print(f"Sys Path: {sys.path}")
print(f"Python Version: {sys.version}")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import racplusplus
import scipy.sparse as sp

print("\nSee how it performs with a connectivity matrix")

# Set up matrix by size and density
rows = 10000
cols = 10000
density = 0.1

# Number of ones and zeros
num_ones = int(rows * cols * density)

# Generate random indices for ones
one_indices = np.random.choice(rows * cols, num_ones, replace=False)

# Create the sparse matrix
data = np.ones(num_ones, dtype=bool)
connectivity_matrix = sp.csc_matrix((data, (np.unravel_index(one_indices, (rows, cols)))), shape=(rows, cols))
print("Done generating sparse unweighted connectivity matrix.\n")

max_merge_distance = .24
batch_size = 1000
no_processors = 0
test_matrix = np.random.random((rows, 768))

print("Running RAC from Python using numpy data matrix and scipy sparse csc connectivity matrix.")
labels = racplusplus.rac(test_matrix, max_merge_distance, connectivity_matrix, batch_size, no_processors)
print(f"Point Cluster Assignments: {labels}")
