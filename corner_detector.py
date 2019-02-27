'''
  File name: corner_detector.py
  Author: Dewang Sultania, Nanda Kishore 
  Date created: 10/31/2018
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line,
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''

from skimage.feature import corner_harris

def corner_detector(img):
    # Your Code Here
    cimg = corner_harris(img, 2, 3, 0.04)
    return cimg
