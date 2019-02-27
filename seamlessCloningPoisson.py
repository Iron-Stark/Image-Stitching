'''
  File name: seamlessCloningPoisson.py
  Author: Nanda Kishore Vasudevan
  Date created:
'''

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from getIndexes import getIndexes
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg

def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):
    # Generate indexes
    indexes = getIndexes(mask, targetImg.shape[0], targetImg.shape[1], offsetX, offsetY)
    num_replacement_pixels = indexes[indexes!=0].size

    # Get A matrix
    coeffA = csr_matrix(getCoefficientMatrix(indexes), dtype=float)

    # Get b vector and solve for x
    x = np.zeros((3, num_replacement_pixels))
    for i in range(3):
        b_i = csr_matrix(getSolutionVect(indexes, sourceImg[:,:,i], targetImg[:,:,i], offsetX, offsetY).reshape(num_replacement_pixels, 1))
        x[i,:] = spsolve(coeffA, b_i)
    x[x<0] = 0
    x[x>255] = 255

    # Combine the three channels
    resultImg = reconstructImg(indexes, x[0,:], x[1,:], x[2,:], targetImg)
    return resultImg
