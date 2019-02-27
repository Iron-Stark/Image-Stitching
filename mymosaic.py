'''
  File name: mymosaic.py
  Author: Dewang Sultania, Nanda Kishore 
  Date created: 10/31/2018
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values.
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

import numpy as np
from matplotlib import pyplot as plt

from seamlessCloningPoisson import seamlessCloningPoisson
from helper import Helper

ALPHA_BLENDING = 0
GRADIENT_BLENDING = 1

def mymosaic(img_input):

    img_left = img_input[0]
    img_center = img_input[1]
    img_right = img_input[2]
    mode = ALPHA_BLENDING
    helper_obj = Helper(img_left, img_center, img_right, mode)
    img_left_stitched, paddings_left, img_right_stitched, paddings_right = \
    helper_obj.execute()

    abs_top = min(paddings_left[0], paddings_right[0])
    abs_bottom = max(paddings_left[1], paddings_right[1])
    height = np.abs(abs_bottom - abs_top)
    abs_left = min(paddings_left[2], paddings_right[2])
    abs_right = max(paddings_left[3], paddings_right[3])
    width = np.abs(abs_right - abs_left)

    final_img_stitched_left = np.zeros((height, width, 3))
    final_img_stitched_left[paddings_left[0]-abs_top:paddings_left[1]-abs_top, \
                            paddings_left[2]-abs_left:paddings_left[3]-abs_left, :] = img_left_stitched


    final_img_stitched_right = np.zeros((height, width, 3))
    final_img_stitched_right[paddings_right[0]-abs_top:paddings_right[1]-abs_top, \
                            paddings_right[2]-abs_left:paddings_right[3]-abs_left, :] = img_right_stitched


    overlapping_indices = np.logical_and([final_img_stitched_left != 0], \
    [final_img_stitched_right != 0])[0]
    if mode == 1:
        mask = overlapping_indices[:,:,0]
        src_img = final_img_stitched_left.copy()
        target_img = final_img_stitched_left + final_img_stitched_right
        target_img[overlapping_indices] = 0
        img_mosaic = np.uint8(seamlessCloningPoisson(src_img, target_img, mask, 0, 0))
    else:
        final_img_stitched_right[overlapping_indices] = final_img_stitched_right\
        [overlapping_indices]/2
        final_img_stitched_left[overlapping_indices] = final_img_stitched_left\
        [overlapping_indices]/2
        img_mosaic = np.uint8((final_img_stitched_right + final_img_stitched_left))

    return img_mosaic
