from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from est_homography import est_homography
from ransac_est_homography import ransac_est_homography
from helper import Helper
from mymosaic import mymosaic

def main():
    folder = 'test_img/'
    filename = '1Hill.JPG'
    img_left = Image.open(folder+filename).convert('RGB')

    img_left = np.array(img_left)

    filename = '2Hill.JPG'
    img_center = Image.open(folder+filename).convert('RGB')

    img_center = np.array(img_center)

    filename = '3Hill.JPG'
    img_right = Image.open(folder+filename).convert('RGB')
    img_right = np.array(img_right)

    imgs = np.array(np.zeros((3)), dtype=object)
    imgs[0] = img_left
    imgs[1] = img_center
    imgs[2] = img_right

    img_mosaic = mymosaic(imgs)
    plt.title('Final image')
    plt.imshow(img_mosaic)
    plt.show()

if __name__=="__main__":
    main()
