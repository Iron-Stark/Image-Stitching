'''
  File name: getCoefficientMatrix.py
  Author: Nanda Kishore Vasudevan
  Date created:
'''

import numpy as np
from scipy.sparse import lil_matrix

def getCoefficientMatrix(indexes):
    # Pad indexes to account for corner cases
    indices = np.zeros((indexes.shape[0]+2, indexes.shape[1]+2))
    indices[1:-1, 1:-1] = indexes.copy()

    # Set up the variable
    targetH, targetW = indices.shape
    num_replacement_pixels = indices[indices!=0].size
    coeffA = lil_matrix((num_replacement_pixels, num_replacement_pixels+1), dtype=float)
    coeffA[np.arange(num_replacement_pixels), np.arange(1, num_replacement_pixels+1)] = 4

    # Find indices for our RoI
    i, j = np.meshgrid(range(targetH), range(targetW), indexing='ij')
    i_not0, j_not0 = i[indices != 0], j[indices!= 0]
    row_index, _ = np.meshgrid(range(num_replacement_pixels), range(4), indexing='ij')


    # Find neighbors of all pixels in our RoI
    neighbors = np.zeros((num_replacement_pixels, 4))
    neighbors[:, 0] = indices[i_not0, j_not0-1]
    neighbors[:, 1] = indices[i_not0, j_not0+1]
    neighbors[:, 2] = indices[i_not0-1, j_not0]
    neighbors[:, 3] = indices[i_not0+1, j_not0]
    neighbors = np.array(neighbors, dtype=np.uint32)

    # Make the values of neighbors -1 in coeffA
    coeffA[row_index, neighbors] = -1
    coeffA = coeffA[:, 1:]

    assert(coeffA.shape == (num_replacement_pixels, num_replacement_pixels))
    return coeffA
