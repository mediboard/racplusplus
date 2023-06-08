import sys
import os
import pybind11
# Append the parent directory of racplusplus package to system path
sys.path.append(os.path.join(os.path.abspath(__file__), "..", "..", "build"))
sys.path.append("/Users/danielfrees/Desktop/racplusplus/build")
print(f"Sys Path: {sys.path}")
print(f"Python Version: {sys.version}")
import racplusplus
import numpy as np

racplusplus.stupid_pybind_test()
racplusplus.test_rac()

#--------Test Actual RAC Endpoint------------
# Set the random seed
np.random.seed(10)
# Generate a random matrix of size 10,000 x 768
test_matrix = np.random.random((10000, 768))
connectivity_matrix = np.empty(0, dtype=bool)
min_distance = .035
batch_size = 1000
no_processors = 0
no_points = 10000

racplusplus.rac(test_matrix, min_distance, batch_size, connectivity_matrix, no_processors, no_points)