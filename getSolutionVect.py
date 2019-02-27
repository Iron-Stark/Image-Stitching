'''
  File name: getSolutionVect.py
  Author: Nanda Kishore Vasudevan
  Date created:
'''

from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt

def getSolutionVect(indexes, source, target, offsetX, offsetY):
    # Initialize variables
    targetH, targetW = target.shape
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    convolved_full = np.zeros((targetH, targetW))

    # Get Laplacian of image
    convolved_full[offsetY:offsetY+source.shape[0], \
    offsetX:offsetX+source.shape[1]] = convolve2d(source, kernel, \
    mode='same', boundary='symm')
    SolVectorb = convolved_full[indexes!=0]

    # Pad image to account for corner cases
    indices = np.zeros((indexes.shape[0]+2, indexes.shape[1]+2))
    indices[1:-1, 1:-1] = indexes.copy()

    # Get indexes for locations in our RoI
    i, j = np.meshgrid(range(targetH+2), range(targetW+2), indexing='ij')
    i_not0, j_not0 = i[indices != 0], j[indices!= 0]

    # Make values in RoI 0 in target image
    masked_target = np.zeros((targetH+2, targetW+2))
    masked_target[1:-1, 1:-1] = target.copy()
    masked_target[indices!=0] = 0

    # Find neighbors for all pixels in RoI
    neighbors = np.zeros((indices[indices!=0].size, 4))
    neighbors[:, 0] = masked_target[i_not0, j_not0-1]
    neighbors[:, 1] = masked_target[i_not0, j_not0+1]
    neighbors[:, 2] = masked_target[i_not0-1, j_not0]
    neighbors[:, 3] = masked_target[i_not0+1, j_not0]
    neighbors = np.array(neighbors, dtype=np.uint32)

    # Sum all neighbor values and the convolved image values
    SolVectorb += np.sum(neighbors, axis=1, dtype=np.uint32)

    SolVectorb = SolVectorb.reshape(1, indexes[indexes!=0].size)
    assert(SolVectorb.shape == (1, indexes[indexes!=0].size))
    return SolVectorb
