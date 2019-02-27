'''
  File name: getIndexes.py
  Author: Nanda Kishore Vasudevan
  Date created:
'''

import numpy as np

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
    # Create variables
    indexes = np.zeros((targetH, targetW))
    indexes[offsetY:offsetY+mask.shape[0], offsetX:offsetX+mask.shape[1]] = \
    mask.copy()

    # Index the pixels that are 1 in the mask
    indexes[indexes!=0] = np.arange(1, indexes[indexes!=0].size+1)
    indexes = np.array(indexes, dtype=np.uint32)

    assert(indexes[indexes!=0].size == indexes.max())
    assert(indexes.shape == targetH, targetW)
    return indexes
