'''
  File name: feat_match.py
  Author: Dewang Sultania, Nanda Kishore 
  Date created: 10/31/2018
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour.
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

import numpy as np
from annoy import AnnoyIndex

def feat_match(descs1, descs2):
    # Your Code Here
    tree = AnnoyIndex(64)
    for index in range(descs2.shape[1]):
        tree.add_item(index, descs2[:,index])
    tree.build(10)
    match = np.zeros((descs1.shape[1], 1))
    distances = np.zeros((descs1.shape[1], 2))
    for index in range(descs1.shape[1]):
        ind, dis = tree.get_nns_by_vector(descs1[:,index], 2, include_distances=True)
        match[index] = np.array(ind[0])
        distances[index, :] = np.array(dis)
    distances = distances[:,0]/distances[:,1]
    match[distances > 0.6] = -1
    return np.int64(match)
