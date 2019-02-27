'''
  File name: feat_desc.py
  Author: Dewang Sultania, Nanda Kishore 
  Date created: 10/31/2018
      '''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature,
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40
    window to have a nice big blurred descriptor.
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator

def GaussianPDF_1D(mu, sigma, length):

  half_len = length / 2

  if np.remainder(length, 2) == 0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)

  ax = ax.reshape([-1, ax.size])
  denominator = sigma * np.sqrt(2 * np.pi)
  nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

  return nominator / denominator

'''
  Generate two dimensional Gaussian distribution
  - input mu: the mean of pdf
  - input sigma: the standard derivation of pdf
  - input row: length in row axis
  - input column: length in column axis
  - output: a 2D matrix represents two dimensional Gaussian distribution
'''
def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()
  return convolve2d(g_row, g_col, 'full')

def feat_desc(img, x, y):
    # Your Code Here
    padded_img = np.pad(img, ((19, 20), (19, 20)), mode='symmetric')
    kernel = GaussianPDF_2D(0,1,5,5)
    gaussian_blur = convolve2d(padded_img, kernel, mode='same', boundary='symm')
    N = x.size
    index_i, index_j = np.meshgrid(np.arange(-18, 19, 5), np.arange(-18, 19, 5), indexing='ij')
    index_i = index_i.flatten()
    index_j = index_j.flatten()
    index_i = np.tile(index_i, (N, 1))
    index_j = np.tile(index_j, (N, 1))
    index_i = np.uint32(index_i + y.reshape(N, 1) + 19)
    index_j = np.uint32(index_j + x.reshape(N, 1) + 19)
    descs = gaussian_blur[index_i, index_j]
    descs = (descs - np.mean(descs))/np.std(descs)
    return descs.T
