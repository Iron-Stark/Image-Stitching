'''
  File name: ransac_est_homography.py
  Author: Dewang Sultania, Nanda Kishore 
  Date created: 10/31/2018
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image.
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from est_homography import est_homography

def ransac_est_homography(x1, y1, x2, y2, thresh):
    # Your Code Here
    max_inliers = 0
    N = x1.shape[0]
    if(N<4):
        print("There are less than four matches")
        exit()
    for _ in range(1000):
        indices = np.random.randint(0, N, size=4)
        x = x1[indices]
        y = y1[indices]
        X = x2[indices]
        Y = y2[indices]
        curr_H = est_homography(x, y, X, Y)
        img1_features = np.concatenate((x1.reshape(1,N), y1.reshape(1,N), np.ones((1, N))), axis=0)
        img2_est_features = np.dot(curr_H, img1_features)
        img2_est_features = img2_est_features[:2]/img2_est_features[2,:]
        img2_features = np.concatenate((x2.reshape(1,N), y2.reshape(1,N)), axis=0)
        errors = np.sqrt(np.sum((img2_features - img2_est_features)**2, axis=0))
        inlier_indices = np.argwhere(errors < thresh)
        inliers = errors[errors < thresh].size
        if inliers > max_inliers:
            max_inliers = inliers
            inlier_ind = inlier_indices
            H = curr_H
    return H, inlier_ind
