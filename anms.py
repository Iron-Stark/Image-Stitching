'''
  File name: anms.py
  Author: Dewang Sultania, Nanda Kishore
  Date created: 10/31/2018
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np
import matplotlib.pyplot as plt

def anms(cimg, max_pts):

    thresh = 0.35

    threshold = np.amin(cimg) + thresh*abs(np.amax(cimg) - np.amin(cimg))
    cimg_thresh = cimg.copy()
    cimg_thresh[cimg > threshold] = 1
    cimg_thresh[cimg_thresh != 1] = 0

    total_features = cimg_thresh[cimg_thresh == 1].size
    if total_features < max_pts:
        print('Please decrease the threshold, there are too few points')
        exit()
    elif total_features > 35000:
        print('Please increase the threshold, there are too many points')
        exit()

    i, j = np.meshgrid(range(cimg_thresh.shape[0]), range(cimg_thresh.shape[1]), indexing='ij')
    i, j = i.astype(float), j.astype(float)
    index_i = i[cimg_thresh == 1].reshape(total_features, 1)
    index_j = j[cimg_thresh == 1].reshape(total_features, 1)
    intensity = cimg[cimg_thresh == 1].reshape(total_features, 1)

    distance_map = (index_i - np.transpose(index_i))**2 + (index_j - np.transpose(index_j))**2

    binary_map = intensity < 0.9*np.transpose(intensity)

    min_radius = binary_map*distance_map
    min_radius[min_radius == 0] = np.inf
    min_radius = np.amin(min_radius, axis=1)
    sorted_indices = np.argsort(min_radius)[::-1]

    min_radius = min_radius[sorted_indices]
    index_i = index_i[sorted_indices]
    index_j = index_j[sorted_indices]

    x = index_j[:max_pts]
    y = index_i[:max_pts]
    rmax = min_radius[max_pts-1]
    return x, y, rmax
