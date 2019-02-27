'''
  File name: reconstructImg.py
  Author: Nanda Kishore Vasudevan
  Date created:
'''

import numpy as np

def reconstructImg(indexes, red, green, blue, targetImg):
    resultImg = targetImg.copy()

    # Modify the pixels in RoI of target image
    resultImg[:,:,0][indexes!=0] = red
    resultImg[:,:,1][indexes!=0] = green
    resultImg[:,:,2][indexes!=0] = blue

    assert(resultImg.shape == (targetImg.shape[0], targetImg.shape[1], targetImg.shape[2]))
    return resultImg
